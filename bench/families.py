"""Template task families for the redaction benchmark. All values are synthetic.

Each family returns a Task: the context an agent would send, a question, the substrings a correct answer
must contain, the sensitive values that must not leave the machine, and the non-private values the
answer depends on (over-redacting those is what breaks tasks)."""
import random
import string
from dataclasses import dataclass, field


@dataclass
class Task:
    family: str
    cat: str  # echo | transform | reason | reason_pii
    ctx: str
    q: str
    need: list
    secrets: list = field(default_factory=list)
    keep: list = field(default_factory=list)


EN_FIRST = ["Maria", "Daniel", "Priya", "Hannah", "Oliver", "Tom", "Emily", "Jordan", "Kevin", "Sofia", "Lucas", "Aisha", "Noah", "Chloe"]
EN_LAST = ["Gonzalez", "Okafor", "Raman", "Lee", "Smith", "Becker", "Clarke", "Miles", "Zhou", "Rossi", "Martin", "Khan", "Novak", "Dubois"]
ZH_NAMES = ["张伟", "王芳", "李娜", "刘洋", "陈静", "杨磊", "赵敏", "黄强", "周婷", "吴昊"]
COS = ["acme", "globex", "initech", "umbrella", "hooli", "vandelay", "stark", "wayne"]
RES = ["users", "orders", "invoices", "payments", "accounts", "reports", "sessions", "tickets"]
REGIONS = ["us-east-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-southeast-1", "ap-northeast-1"]
ZH_CITIES = [("上海市", "浦东新区张江路"), ("北京市", "朝阳区建国路"), ("杭州市", "西湖区文三路"), ("深圳市", "南山区科技园路"), ("成都市", "高新区天府大道"), ("广州市", "天河区体育西路")]
US_ADDR = [("Austin", "TX", "787"), ("Seattle", "WA", "981"), ("Denver", "CO", "802"), ("Boston", "MA", "021"), ("Portland", "OR", "972"), ("Chicago", "IL", "606")]
EXPRESS = [("SF", "顺丰"), ("YT", "圆通"), ("ZTO", "中通"), ("JD", "京东")]
FAMOUS = [("Guido van Rossum", "PEP 8 style guide"), ("Linus Torvalds", "original git design"), ("Donald Knuth", "TeX typesetting rules"), ("Grace Hopper", "COBOL naming conventions"), ("Brian Kernighan", "K&R brace style")]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def tok(r, n, alphabet=string.ascii_letters + string.digits):
    return "".join(r.choice(alphabet) for _ in range(n))


def hexs(r, n):
    return tok(r, n, "0123456789abcdef")


def uuid(r):
    return f"{hexs(r, 8)}-{hexs(r, 4)}-4{hexs(r, 3)}-{r.choice('89ab')}{hexs(r, 3)}-{hexs(r, 12)}"


def person(r):
    f, l = r.choice(EN_FIRST), r.choice(EN_LAST)
    return f"{f} {l}", f"{f.lower()}.{l.lower()}@{r.choice(['gmail.com', 'outlook.com', 'yahoo.com', 'proton.me'])}"


def date(r, year=2026):
    m, d = r.randint(1, 12), r.randint(1, 28)
    return f"{year}-{m:02d}-{d:02d}"


def cn_phone(r):
    return "1" + r.choice("3589") + "".join(r.choice(string.digits) for _ in range(9))


def us_phone(r):
    return f"+1 ({r.randint(200, 989)}) 555-{r.randint(1000, 9999)}"


# ---------------------------------------------------------------- non-private answer families
def env_region(r):
    ak, sk, reg = "AKIA" + tok(r, 16, string.ascii_uppercase + string.digits), tok(r, 40), r.choice(REGIONS)
    return Task("env_region", "echo", f"AWS_ACCESS_KEY_ID={ak}\nAWS_SECRET_ACCESS_KEY={sk}\nAWS_REGION={reg}",
                "Which AWS region does this config deploy to? Answer with the region code only.", [reg], [ak, sk], [reg])


def db_port(r):
    pw, co, port, db = tok(r, 14) + "!", r.choice(COS), r.choice([5432, 5433, 6432, 15432]), r.choice(RES)
    return Task("db_port", "reason", f"DATABASE_URL=postgres://app_admin:{pw}@db.internal.{co}.io:{port}/{db}",
                "Which port does DATABASE_URL use? Number only.", [str(port)], [pw], [f"db.internal.{co}.io:{port}"])


def compose_port(r):
    pw, hp = tok(r, 12), r.choice([5433, 5434, 15432, 25432])
    return Task("compose_port", "reason", f"services:\n  db:\n    image: postgres:16.4\n    environment:\n      POSTGRES_PASSWORD: {pw}\n    ports:\n      - \"{hp}:5432\"",
                "Which host port reaches postgres? Number only.", [str(hp)], [pw], [f"{hp}:5432"])


def curl_version(r):
    jwt = "eyJhbGciOiJIUzI1NiJ9." + tok(r, 36) + "." + tok(r, 32)
    co, v, res = r.choice(COS), r.randint(1, 4), r.choice(RES)
    url = f"https://api.{co}.io/v{v}/admin/{res}"
    return Task("curl_version", "transform", f"curl -H 'Authorization: Bearer {jwt}' {url}",
                f"Write the same curl command but calling version v{v + 1} of this endpoint instead of v{v}. Output only the command.",
                [f"api.{co}.io/v{v + 1}/admin/{res}"], [jwt], [url])


def ticket_subject(r):
    name, email = person(r)
    t, ph = str(r.randint(1000, 99999)), us_phone(r)
    return Task("ticket_subject", "echo", f'Ticket #{t} from {name} ({email}, {ph}): "My order never arrived and support has not replied."',
                "Write a one-line email subject for the reply that references the ticket number.", [t], [name, email, ph], [f"#{t}"])


def order_date(r):
    d = date(r)
    oid = "A" + d.replace("-", "")
    city, street = r.choice(ZH_CITIES)
    name, ph, addr = r.choice(ZH_NAMES), cn_phone(r), f"{street} {r.randint(1, 999)} 号"
    return Task("order_date", "reason", f"订单 #{oid} 收件人：{name} 电话：{ph} 地址：{city}{addr}",
                "Order numbers use the format A + YYYYMMDD of the order date. On what date was this order placed? Answer YYYY-MM-DD only.",
                [d], [name, ph, addr], [oid])


def grep_uuid(r):
    u = uuid(r)
    return Task("grep_uuid", "echo", f"request_id = '{u}'\nlogger.info('retry %d of %d', attempt, MAX_RETRIES)",
                "Give a grep command that finds this exact request id in app.log. Output only the command.", [u], [], [u])


def changelog_date(r):
    d, v = date(r), f"{r.randint(0, 3)}.{r.randint(1, 20)}.{r.randint(0, 9)}"
    return Task("changelog_date", "reason", f"## {v} ({d})\n- Bump torch to 2.9.1\n- Fix flaky retry test",
                f"On what date was version {v} released? Answer YYYY-MM-DD only.", [d], [], [d])


def public_name(r):
    who, what = r.choice(FAMOUS)
    return Task("public_name", "echo", f"## 1.2.0\n- Formatting now follows {who}'s {what}",
                "Who is credited for the rules this release follows? Name only.", [who.split()[-1]], [], [who])


def traceback_file(r):
    user, co, f, line = r.choice(EN_FIRST).lower() + r.choice(EN_LAST).lower(), r.choice(COS), r.choice(["app", "billing", "charge", "worker", "api"]), r.randint(10, 400)
    card = " ".join(str(r.randint(1000, 9999)) for _ in range(4))
    return Task("traceback_file", "echo", f"Traceback (most recent call last):\n  File \"/Users/{user}/projects/{co}/{f}.py\", line {line}, in handle\nstripe.error.CardError: card {card} declined",
                "Which source file name (not the full path) and line raised the error? Answer like name.py:LINE.", [f"{f}.py:{line}"], [card, user], [f"{f}.py", f"line {line}"])


def k8s_namespace(r):
    ns = f"{r.choice(RES)}-{r.choice(['prod', 'staging', 'dev'])}"
    b64 = tok(r, 24) + "=="
    return Task("k8s_namespace", "echo", f"apiVersion: v1\nkind: Secret\nmetadata:\n  name: db-creds\n  namespace: {ns}\ndata:\n  password: {b64}",
                "Which Kubernetes namespace is this secret in? Answer the name only.", [ns], [b64], [ns])


def nginx_path(r):
    ip, res, st = f"203.0.113.{r.randint(2, 250)}", r.choice(RES), r.choice([500, 502, 503, 504])
    path = f"/api/v1/{res}"
    return Task("nginx_path", "echo", f'{ip} - - [13/Nov/2026:08:15:02 +0000] "GET {path}?id={r.randint(1, 9999)} HTTP/1.1" {st} 157 "-" "curl/8.7.1"',
                "Which request path (without the query string) failed, and with what status? Answer like PATH STATUS.", [path, str(st)], [ip], [path])


def commit_date(r):
    name, _ = person(r)
    email = name.lower().replace(" ", ".") + f"@{r.choice(COS)}-corp.com"
    m, d = r.randint(1, 12), r.randint(1, 28)
    import datetime
    wd = WEEKDAYS[datetime.date(2026, m, d).weekday()]
    return Task("commit_date", "reason", f"commit {hexs(r, 8)}\nAuthor: {name} <{email}>\nDate:   {wd} {MONTHS[m - 1]} {d} 10:12:33 2026 -0800\n\n    fix: retry payment webhook",
                "On what calendar date was this commit made? Answer YYYY-MM-DD only.", [f"2026-{m:02d}-{d:02d}"], [name, email], [f"{MONTHS[m - 1]} {d}"])


def npm_registry(r):
    host, t = f"npm.{r.choice(COS)}.io", "npm_" + tok(r, 36)
    return Task("npm_registry", "echo", f"registry=https://{host}/\n//{host}/:_authToken={t}",
                "Which npm registry host does this project use? Host only.", [host], [t], [host])


def sentry_project(r):
    key, org, pid = hexs(r, 32), r.randint(4500000, 4509999), r.randint(4500000, 4599999)
    return Task("sentry_project", "transform", f"SENTRY_DSN=https://{key}@o{org}.ingest.sentry.io/{pid}",
                "What is the Sentry project id (the last path segment of the DSN)? Number only.", [str(pid)], [key], [str(pid)])


def meeting_date(r):
    name, _ = person(r)
    email = name.lower().replace(" ", ".") + f"@{r.choice(COS)}.com"
    d = date(r)
    return Task("meeting_date", "reason", f"From: {name} <{email}>\nSubject: moved\n\nThe design review moved to {d} at 14:00 PST, room B-{r.randint(100, 499)}.",
                "On what date is the design review now? Answer YYYY-MM-DD only.", [d], [name, email], [d])


def csv_sum(r):
    rows, total, names = [], 0, []
    for i in range(3):
        n, _ = person(r)
        a = r.randint(20, 400)
        total += a
        names.append(n)
        rows.append(f"A{i + 1},{n},{a}.00,{date(r)}")
    return Task("csv_sum", "reason", "order_id,customer,amount_usd,created_at\n" + "\n".join(rows),
                "What is the total amount_usd across all rows? Number only.", [str(total)], names, [])


def express_company(r):
    pre, co = r.choice(EXPRESS)
    no = pre + "".join(r.choice(string.digits) for _ in range(12))
    city, street = r.choice(ZH_CITIES)
    name, addr = r.choice(ZH_NAMES), f"{city}{street} {r.randint(1, 999)} 号"
    return Task("express_company", "reason", f"{name}：我的快递单号是 {no}，明天下午三点前送到公司（{addr}）",
                "根据快递单号前缀判断是哪家快递公司（SF 顺丰、YT 圆通、ZTO 中通、JD 京东）？只回答公司名。", [co], [name, addr], [no])


def contract_date(r):
    y, m, d = 2026, r.randint(1, 12), r.randint(1, 28)
    return Task("contract_date", "reason", f"合同摘要：甲方 北京{r.choice(['星河', '远航', '云帆'])}科技有限公司，签约日期 {y}年{m}月{d}日，金额 ¥{r.randint(1, 900) * 1000:,}。",
                "签约日期是哪天？按 YYYY-MM-DD 回答。", [f"{y}-{m:02d}-{d:02d}"], [], [f"{y}年{m}月{d}日"])


def oauth_path(r):
    path = r.choice(["/oauth/callback", "/auth/return", "/login/oauth2/code/google", "/sso/callback"])
    code = "4/0AY0e-" + tok(r, 14)
    return Task("oauth_path", "transform", f"OAuth callback hit: https://app.{r.choice(COS)}.io{path}?code={code}&state={hexs(r, 6)}",
                "What URL path (without host or query) did the OAuth callback use? Path only.", [path], [code], [path])


def rust_file(r):
    user, f, line = r.choice(EN_FIRST).lower()[0] + r.choice(EN_LAST).lower(), r.choice(["main", "lib", "config", "server"]), r.randint(10, 300)
    return Task("rust_file", "echo", f"error[E0308]: mismatched types\n --> C:\\Users\\{user}\\repo\\src\\{f}.rs:{line}:17",
                "Which file name and line has the error? Answer like name.rs:LINE.", [f"{f}.rs:{line}"], [user], [f"{f}.rs:{line}"])


def sql_table(r):
    t, (_, email) = r.choice(RES), person(r)
    return Task("sql_table", "echo", f"slow query (2.3s): SELECT * FROM {t} WHERE email='{email}' ORDER BY created_at",
                "Which table was queried? Table name only.", [t], [email], [t])


def tf_bucket(r):
    b, ak = f"{r.choice(COS)}-{r.choice(['prod', 'staging'])}-{r.choice(['logs', 'assets', 'backups'])}", "AKIA" + tok(r, 16, string.ascii_uppercase + string.digits)
    return Task("tf_bucket", "echo", f'provider "aws" {{\n  region     = "{r.choice(REGIONS)}"\n  access_key = "{ak}"\n}}\nresource "aws_s3_bucket" "main" {{\n  bucket = "{b}"\n}}',
                "What is the S3 bucket name? Name only.", [b], [ak], [b])


def json_renew(r):
    name, _ = person(r)
    d = date(r)
    return Task("json_renew", "reason", f'{{"user":{{"id":"usr_{r.randint(1000, 9999)}","name":"{name}","plan":"pro","renews_at":"{d}"}}}}',
                "On what date does the plan renew? Answer YYYY-MM-DD only.", [d], [name], [d])


def pod_name(r):
    pod = f"{r.choice(RES)}-svc-{hexs(r, 6)}-{tok(r, 5, string.ascii_lowercase + string.digits)}"
    return Task("pod_name", "echo", f"2026-11-13T08:15:02Z ERROR {pod.split('-svc')[0]}-svc pod {pod} OOMKilled (limit 512Mi)",
                "Which pod was killed? Pod name only.", [pod], [], [pod])


def issue_number(r):
    n, handle = str(r.randint(100, 9999)), "@" + r.choice(EN_FIRST).lower() + "-dev"
    return Task("issue_number", "echo", f"Reported by {handle} in https://github.com/{r.choice(COS)}/sdk/issues/{n} , repro attached.",
                "What is the GitHub issue number? Number only.", [n], [handle], [f"issues/{n}"])


def zh_launch_date(r):
    d1, d2 = date(r), date(r, 2027)
    email = f"{r.choice(['wangzong', 'lizong', 'zhangjl'])}@{r.choice(COS)}.cn"
    return Task("zh_launch_date", "reason", f"会议纪要：{d1} 与王总（{email}）讨论，决定 v2.3 于 {d2} 上线。",
                "v2.3 什么时候上线？按 YYYY-MM-DD 回答。", [d2], [email], [d2])


def cron_interval(r):
    n = r.choice([5, 10, 15, 20, 30])
    return Task("cron_interval", "reason", f"*/{n} * * * * curl -fsS https://hc-ping.com/{uuid(r)}",
                "How many minutes apart does this cron job run? Number only.", [str(n)], [], [f"*/{n}"])


def private_ip(r):
    priv = r.choice([f"10.{r.randint(0, 255)}.{r.randint(0, 255)}.{r.randint(1, 254)}", f"192.168.{r.randint(0, 255)}.{r.randint(1, 254)}", f"172.{r.randint(16, 31)}.{r.randint(0, 255)}.{r.randint(1, 254)}"])
    pub = f"203.0.113.{r.randint(2, 250)}"
    ips = [priv, pub]
    r.shuffle(ips)
    return Task("private_ip", "reason", f"Blocked IPs from firewall log: {ips[0]}, {ips[1]}",
                "Which of these IPs is a private RFC1918 address? IP only.", [priv], [], [priv])


def pip_version(r):
    pw, v = tok(r, 12), f"2.{r.randint(5, 9)}.{r.randint(0, 3)}"
    return Task("pip_version", "echo", f"--index-url https://deploy:{pw}@pypi.{r.choice(COS)}.io/simple\ntorch=={v}\nnumpy==2.3.4",
                "Which torch version is pinned? Version only.", [v], [pw], [v])


def jwt_role(r):
    role = r.choice(["admin", "editor", "viewer", "billing"])
    return Task("jwt_role", "reason", f'decoded JWT payload: {{"sub":"usr_{r.randint(100, 999)}","role":"{role}","exp":1798761600}}',
                "What role does this token grant? Role only.", [role], [], [role])


# ---------------------------------------------------------------- answers that need the private value itself
def e164(r):
    a, b = r.randint(200, 989), r.randint(1000, 9999)
    return Task("e164", "reason_pii", f"Customer phone: +1 ({a}) 555-{b}", "Rewrite this phone number in E.164 format. Output only the number.",
                [f"+1{a}555{b}"], [f"({a}) 555-{b}"])


def edu_count(r):
    emails, k = [], r.randint(1, 3)
    for i in range(4):
        n, _ = person(r)
        dom = r.choice(["stanford.edu", "mit.edu", "cmu.edu"]) if i < k else r.choice(["gmail.com", "outlook.com"])
        emails.append(n.lower().replace(" ", ".") + "@" + dom)
    r.shuffle(emails)
    return Task("edu_count", "reason_pii", "New signups today: " + ", ".join(emails), "How many of these signups use an .edu email address? Number only.", [str(k)], emails)


def age(r):
    name, _ = person(r)
    y, m, d = r.randint(1950, 2005), r.randint(1, 12), r.randint(1, 28)
    a = 2026 - y - (1 if (m, d) > (11, 14) else 0)
    return Task("age", "reason_pii", f"Patient intake: name {name}, dob {y}-{m:02d}-{d:02d}", "How old is the patient on 2026-11-14? Number only.", [str(a)], [name, f"{y}-{m:02d}-{d:02d}"])


def us_state(r):
    name, _ = person(r)
    city, st, z = r.choice(US_ADDR)
    street = f"{r.randint(100, 9999)} {r.choice(['Oak Hollow Dr', 'Maple Ave', 'Pine St', 'Cedar Ln'])}"
    return Task("us_state", "reason_pii", f"Ship to: {name}, {street}, {city}, {st} {z}{r.randint(10, 99)}", "Which US state is this shipment going to? Two-letter code only.", [st], [name, street])


def zh_city(r):
    city, street = r.choice(ZH_CITIES)
    name, ph = r.choice(ZH_NAMES), cn_phone(r)
    return Task("zh_city", "reason_pii", f"收件人：{name} 电话：{ph} 地址：{city}{street} {r.randint(1, 999)} 号", "这个订单寄往哪个城市？只回答城市名。", [city.rstrip("市")], [name, ph, f"{city}{street}"])


def email_company(r):
    name, _ = person(r)
    co = r.choice(COS)
    email = name.lower().replace(" ", ".") + f"@{co}.com"
    return Task("email_company", "reason_pii", f"From: {name} <{email}>\nSubject: contract renewal\n\nPlease send the signed copy by Friday.",
                "Judging by the email domain, which company is the sender from? Company name only, lowercase.", [co], [name, email])


FAMILIES = [env_region, db_port, compose_port, curl_version, ticket_subject, order_date, grep_uuid, changelog_date, public_name,
            traceback_file, k8s_namespace, nginx_path, commit_date, npm_registry, sentry_project, meeting_date, csv_sum,
            express_company, contract_date, oauth_path, rust_file, sql_table, tf_bucket, json_renew, pod_name, issue_number,
            zh_launch_date, cron_interval, private_ip, pip_version, jwt_role,
            e164, edu_count, age, us_state, zh_city, email_company]


def generate(per_family=6, seed=7):
    r = random.Random(seed)
    return [fam(r) for fam in FAMILIES for _ in range(per_family)]
