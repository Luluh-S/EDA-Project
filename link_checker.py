
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).parent / "models"

# Free website builders and hosts: anyone can make "something.web.app" in a minute.
FREE_HOSTS = {
    "web.app", "firebaseapp.com", "github.io", "netlify.app", "vercel.app", "pages.dev", "workers.dev",
    "weebly.com", "weeblysite.com", "wixsite.com", "blogspot.com", "000webhostapp.com", "herokuapp.com",
    "glitch.me", "repl.co", "replit.app", "webflow.io", "square.site", "godaddysites.com", "ipfs.io",
    "azurewebsites.net", "appspot.com", "ngrok.io", "duckdns.org", "backname.io", "r2.dev", "onrender.com",
    "framer.app", "mystrikingly.com", "wordpress.com", "sites.google.com", "notion.site",
}
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "cutt.ly", "is.gd", "rb.gy", "shorturl.at", "t.ly", "goo.gl", "ow.ly"}
BRANDS = ["paypal", "apple", "microsoft", "google", "amazon", "netflix", "facebook", "instagram", "whatsapp",
          "telegram", "linkedin", "icloud", "outlook", "office365", "dropbox", "docusign", "adobe", "yahoo",
          "binance", "coinbase", "metamask", "ebay", "dhl", "fedex", "usps", "aramex", "absher", "alrajhi", "stcpay"]
# digits and letter pairs that phishing links use to fake a brand: paypa1, amaz0n, arnazon
LOOKALIKE = str.maketrans("0134578", "oleastb")

# Big platforms where anyone can upload a page: cloud storage, file sharing, form and shop builders.
# They're on the popular list, but a page there could have been made by anyone.
OPEN_PLATFORMS = {
    "amazonaws.com", "cloudfront.net", "googleapis.com", "googleusercontent.com", "windows.net", "azureedge.net",
    "digitaloceanspaces.com", "akamaized.net", "b-cdn.net", "sharepoint.com", "dropbox.com", "box.com", "1drv.ms",
    "mediafire.com", "myshopify.com", "linktr.ee", "typeform.com", "jotform.com", "telegra.ph", "fly.dev", "site123.me",
    "myqcloud.com", "myhuaweicloud.com", "backblazeb2.com", "wasabisys.com", "cloudwaysapps.com", "amplifyapp.com",
    "ondigitalocean.app", "fastly.net", "page.link", "filesusr.com", "jimdofree.com",
}
# kept for examples in documentation, never a real service
RESERVED = {"example.com", "example.net", "example.org"}
# words phishing links put in front of the real name or after it, where the model doesn't look
BAIT_WORDS = ["login", "log-in", "signin", "sign-in", "verify", "verification", "account", "secure", "security",
              "update", "confirm", "password", "unlock", "suspend", "billing", "wallet", "recover", "webscr"]

# Saudi sites the checker trusts. Only government bodies can get a .gov.sa name and only universities
# and schools an .edu.sa one, so those endings are trusted as a whole. The banks were checked one by one.
TRUSTED_ENDINGS = {"gov.sa", "edu.sa"}
TRUSTED_SITES = {
    "alrajhibank.com.sa", "alahli.com", "riyadbank.com", "sab.com", "bsf.sa", "anb.com.sa", "alinma.com",
    "alinmabank.com", "bankalbilad.com", "absher.sa", "stc.com.sa",
}

# endings made of two parts, like com.sa or co.uk
TWO_PART = {"co", "com", "org", "net", "ac", "gov", "edu"}

SCHEME = r"^[a-zA-Z][a-zA-Z0-9+.-]*://"

LINK_FEATURES = ["name_length", "name_digits", "name_hyphens", "name_vowel_ratio", "tld_length", "tld_legit_prob",
                 "on_free_host", "name_entropy", "name_consonant_run", "name_digit_switches", "brand_in_name",
                 "brand_lookalike"]


def split_link(url):
   #Break a link into its parts.
    url = url.strip()
    if not re.match(SCHEME, url):
        url = "http://" + url
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    host = host.removeprefix("www.")

    labels = host.split(".")
    if len(labels) < 2 or re.fullmatch(r"[\d.]+", host):      # a single word or an IP address
        return {"host": host, "subdomain": "", "name": host, "tld": "", "registered": host}

    n = 2 if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in TWO_PART else 1
    tld = ".".join(labels[-n:])
    name = labels[-n - 1]
    registered = f"{name}.{tld}"
    subdomain = ".".join(labels[:-n - 1])

    # on a free host the owner picks the part in front, e.g. "alrajhi-login" in alrajhi-login.web.app
    if registered in FREE_HOSTS and subdomain:
        name = subdomain.split(".")[-1]
    return {"host": host, "subdomain": subdomain, "name": name, "tld": tld, "registered": registered}


def edit_distance(a, b):
    #How many letters you have to add, remove, change or swap with a neighbour to turn a into b.
    before, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i]
        for j in range(1, len(b) + 1):
            d = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d = min(d, before[j - 2] + 1)
            cur.append(d)
        before, prev = prev, cur
    return prev[-1]


def lookalike_brand(name):
    #The brand a name imitates, like 'paypal' for paypa1 or 'facebook' for facebok. None if it imitates nothing.

    for part in dict.fromkeys([name, *name.split("-")]):
        if not part or part in BRANDS:
            continue
        swapped = part.translate(LOOKALIKE).replace("rn", "m").replace("vv", "w")
        for brand in BRANDS:
            if len(brand) >= 4 and brand in swapped and brand not in part:
                return brand
            # one letter off for longer brands, two for really long ones. The first letter has to match,
            # otherwise ordinary words like finance (binance) and cloud (icloud) count too.
            if len(brand) >= 6 and part[0] == brand[0] and abs(len(part) - len(brand)) <= 2 and \
                    edit_distance(part, brand) <= (1 if len(brand) < 8 else 2):
                return brand
    return None


def brand_inside(name):
    #A brand that is part of the name without being the whole name, like 'paypal' in paypal-secure.
    return next((b for b in BRANDS if len(b) >= 4 and b in name and b != name), None)


def link_features(url, tld_probs, default_prob=0.5):
    #The 12 numbers the link model uses. tld_probs maps each ending to its TLDLegitimateProb.
    p = split_link(url)
    name = p["name"]
    letters = [c for c in name if c.isalpha()]
    chars = [c for c in name if c.isalnum()]
    ending = p["tld"].split(".")[-1]
    counts = Counter(chars).values()
    return {
        "name_length": len(name),
        "name_digits": sum(c.isdigit() for c in name),
        "name_hyphens": name.count("-"),
        "name_vowel_ratio": sum(c in "aeiou" for c in letters) / len(letters) if letters else 0.0,
        "tld_length": len(ending),
        "tld_legit_prob": tld_probs.get(ending, default_prob),
        "on_free_host": int(p["registered"] in FREE_HOSTS),
        # how random the name looks: mixed-up characters, long runs of consonants, digits and letters swapping
        "name_entropy": -sum(n / len(chars) * math.log2(n / len(chars)) for n in counts),
        "name_consonant_run": max(map(len, re.findall(r"[bcdfghjklmnpqrstvwxz]+", name)), default=0),
        "name_digit_switches": sum(a.isdigit() != b.isdigit() for a, b in zip(chars, chars[1:])),
        # how close the name is to a famous brand
        "brand_in_name": int(brand_inside(name) is not None),
        "brand_lookalike": int(lookalike_brand(name) is not None),
    }


def link_text(url):
    #Name + ending as text, e.g. 'alrajhi-secure .app'. The model reads its letter patterns.
    p = split_link(url)
    return f"{p['name']} .{p['tld'].split('.')[-1]}"


def link_frame(urls, tld_probs):
    #The model's input for a list of links: the 12 numbers plus the text column.
    X = pd.DataFrame([link_features(u, tld_probs) for u in urls])
    X["text"] = [link_text(u) for u in urls]
    return X


def bait_words(url):
    #Bait words in the parts of a link the model doesn't read: in front of the real name and after it.
    url = url.strip()
    parts = urlsplit(url if re.match(SCHEME, url) else "http://" + url)
    outside = f"{split_link(url)['subdomain']} {parts.path} {parts.query} {parts.fragment}".lower()
    return [w for w in BAIT_WORDS if w in outside]


def red_flags(url, popular=frozenset()):
    #Simple rules for tricks the model can't see. Returns a list of (severity, message).
    p = split_link(url)
    host, sub, reg = p["host"], p["subdomain"], p["registered"]
    is_ip = bool(re.fullmatch(r"[\d.]+", host))
    flags = []

    if is_ip:
        flags.append(("high", "Uses a raw IP address instead of a website name."))
    if "@" in url.split("//")[-1].split("/")[0]:
        flags.append(("high", "Contains '@' in the address. The browser ignores everything before it."))
    if not is_ip and re.search(r"(\d{1,3}[.-]){3}\d{1,3}", host):
        flags.append(("high", "An IP address is hidden inside the website name."))
    if "xn--" in host:
        flags.append(("high", "Uses look-alike letters (punycode) that can imitate a real brand."))

    brand_in_sub = [b for b in BRANDS if b in sub and b not in p["name"]]
    if brand_in_sub:
        flags.append(("high", f"'{brand_in_sub[0]}' comes before the real site name. The site is actually {reg}."))
    fake = lookalike_brand(p["name"])
    # a few real sites are close to a brand by chance (telegraph and telegram), so popular ones are left out
    if fake and reg not in TRUSTED_SITES and reg not in popular:
        flags.append(("high", f"'{p['name']}' looks a lot like '{fake}' but isn't spelled the same. Fake sites copy "
                              "famous names this way."))
    if len([s for s in sub.split(".") if s]) >= 3:
        flags.append(("medium", f"Many subdomains stacked in front of the real name ({reg})."))

    if reg in FREE_HOSTS:
        brand_on_free = [b for b in BRANDS if b in p["name"]]
        if brand_on_free:
            flags.append(("high", f"'{brand_on_free[0]}' on a free website builder ({reg}). Real companies don't do this."))
        else:
            flags.append(("medium", f"Hosted on a free website builder ({reg}). Anyone can make one in minutes."))
    if reg in OPEN_PLATFORMS:
        flags.append(("medium", f"Hosted on {reg}, a platform where anyone can upload a page. The big name doesn't "
                                "mean that company made it."))
    if reg in SHORTENERS:
        flags.append(("medium", "This is a link shortener, so the real destination is hidden."))
    if reg in RESERVED:
        flags.append(("medium", f"{reg} is kept for examples in documentation. No real company sends links to it."))

    words = ", ".join(f"'{w}'" for w in bait_words(url))
    if words and reg in FREE_HOSTS | OPEN_PLATFORMS:
        flags.append(("high", f"Words like {words} on a page anyone could have made."))
    elif words:
        flags.append(("medium" if "," in words else "low",
                      f"Words that phishing links often use: {words}. They're normal on a real login page, "
                      f"so make sure {reg} is the site you expect."))

    if url.strip().lower().startswith("http://"):
        flags.append(("low", "No HTTPS (but HTTPS alone doesn't prove a site is safe)."))
    order = ["high", "medium", "low"]
    return sorted(flags, key=lambda f: order.index(f[0]))


def phishing_score(url, checker):
    """The model's phishing score for a link, from 0 to 1.

    The pipeline gives a raw score and the calibrator turns it into a real chance: of all the training links
    that got about 0.4, about 40% were phishing.
    """
    raw = checker["pipeline"].predict_proba(link_frame([url], checker["tld_probs"]))[:, 1]
    return float(checker["calibrator"].predict(raw)[0])


def load_checker(folder=MODEL_DIR):
    #The link model saved by the model notebook, plus the list of popular websites
    folder = Path(folder)
    checker = joblib.load(folder / "link_model.joblib")
    checker["popular"] = set((folder / "popular_domains_top10k.txt").read_text().split())
    return checker


def is_trusted(url):
    #True for a Saudi government or university site, or one of the Saudi banks and services in TRUSTED_SITES.
    p = split_link(url)
    return p["tld"] in TRUSTED_ENDINGS or p["registered"] in TRUSTED_SITES


def check_link(url, checker):
    #Model score + rules + trusted and popular lists is checked. The link itself is never opened.
    p = split_link(url)
    score = phishing_score(url, checker)
    flags = red_flags(url, checker["popular"])
    high = any(sev == "high" for sev, _ in flags)
    platform = p["registered"] in OPEN_PLATFORMS
    trusted = is_trusted(url)
    popular = (p["registered"] in checker["popular"]
               and p["registered"] not in FREE_HOSTS | SHORTENERS | OPEN_PLATFORMS | RESERVED)
    threshold = checker["threshold"]

    # on an open platform the model only sees the platform's name (amazonaws.com), so its score says nothing
    if trusted and not high:
        verdict = "Trusted Saudi website"
    elif popular and not high:
        verdict = "Known popular website"
    elif high or (score >= threshold and not platform):
        verdict = "Likely phishing"
    elif flags or score >= threshold - 0.15:
        verdict = "Suspicious"
    else:
        verdict = "Looks OK"
    return {"verdict": verdict, "score": score, "threshold": threshold, "registered": p["registered"],
            "trusted": trusted, "popular": popular, "platform": platform, "flags": flags}


def explain_score(url, checker):
    #Letter patterns in the link's name and how much each one pushed the model's score.

    
    features, regression = checker["pipeline"][0], checker["pipeline"][-1]
    names = features.get_feature_names_out()
    x = features.transform(link_frame([url], checker["tld_probs"])).tocsr()
    pushes = {}
    for i, value in zip(x.indices, x.data):
        kind, chunk = names[i].split("__", 1)
        if kind == "letters" and len(chunk.strip()) >= 3:
            pushes[chunk.strip()] = pushes.get(chunk.strip(), 0) + value * regression.coef_[0][i]
    return sorted(pushes.items(), key=lambda kv: kv[1], reverse=True)
