"""Shared pieces: BigCodeBench tasks, the unit-test harness, stats, and the J-lens readout."""
import ast, gzip, json, os, re, subprocess, tempfile
import numpy as np

DATA = os.environ.get("DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
BCB_PY = os.environ.get("BCB_PY", "/opt/bcbenv/bin/python")  # python env with BigCodeBench deps, see setup_bcb.sh
SYSTEM = ("You are a helpful assistant that generates correct Python code for a given coding problem.\n"
          "Your reasoning should be concise and to the point.\n"
          "Please provide a self-contained Python script that is logically correct, syntactically valid, compiles, and follows the expected output format.\n"
          "Format the program in a markdown code block.\n")
LENS_REPO, LENS_FILE = "neuronpedia/jacobian-lens", "qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt"
BAND = list(range(15, 37))  # J-lens workspace band for Qwen3-14B (40 layers)


def path(*p):
    return os.path.join(DATA, *p)


def read_jsonl(p):
    with (gzip.open(p, "rt") if p.endswith(".gz") else open(p)) as f:
        return [json.loads(l) for l in f]


def write_jsonl(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with (gzip.open(p, "wt") if p.endswith(".gz") else open(p, "w")) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_tasks():
    from datasets import load_dataset
    ds = load_dataset("Joschka/bigcodebench", "bcb-test")
    return sorted([dict(r) for r in ds[list(ds.keys())[0]]], key=lambda r: r["task_id"])


def ok_task_ids():
    """Tasks whose canonical solution passes all tests in our harness (222 of 228)."""
    return {r["task_id"] for r in read_jsonl(path("harness.jsonl"))
            if r["status"] == "ok" and r["total"] and r["passed"] == r["total"]}


# ---------------------------------------------------------------- unit-test harness
RUNNER = r'''
import sys, json, io, os, unittest, contextlib
os.environ["MPLBACKEND"]="Agg"
src=open(sys.argv[1]).read(); test=open(sys.argv[2]).read()
ns={"__name__":"__solution__"}
res={"status":"ok","passed":0,"total":None}
out=io.StringIO()
try:
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        exec(compile(src,"solution","exec"),ns)
except BaseException as e:
    res["status"]="exec_error"; res["err"]=repr(e)[:300]
if res["status"]=="ok":
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            exec(compile(test,"test","exec"),ns)
            suite=unittest.TestLoader().loadTestsFromTestCase(ns["TestCases"])
            r=unittest.TestResult(); suite.run(r)
        res["total"]=r.testsRun; res["passed"]=r.testsRun-len(r.failures)-len(r.errors)-len(r.unexpectedSuccesses)
    except BaseException as e:
        res["status"]="test_error"; res["err"]=repr(e)[:300]
print("@@RES@@"+json.dumps(res))
'''


def split_visible(text, think_off):
    """(reasoning, visible answer). With thinking off the model sometimes reopens a think block."""
    if think_off:
        return tuple(text.rsplit("</think>", 1)) if "</think>" in text else ("", text)
    return tuple(text.rsplit("</think>", 1)) if "</think>" in text else (text, "")


def extract_code(vis):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", vis, re.S)
    if blocks:
        return blocks[-1]
    m = re.search(r"```(?:python|py)?\s*\n(.*)$", vis, re.S)
    return m.group(1) if m else None


def n_tests(test_src):
    try:
        for node in ast.walk(ast.parse(test_src)):
            if isinstance(node, ast.ClassDef) and node.name == "TestCases":
                return sum(isinstance(b, ast.FunctionDef) and b.name.startswith("test") for b in node.body)
    except Exception:
        pass
    return None


def run_tests(code, test):
    nt = n_tests(test)
    if not code:
        return {"status": "no_code", "passed": 0, "total": nt}
    with tempfile.TemporaryDirectory() as d:
        for name, src in (("s.py", code), ("t.py", test), ("r.py", RUNNER)):
            open(os.path.join(d, name), "w").write(src)
        try:
            p = subprocess.run(["bash", "-c", f"ulimit -v 25000000; cd {d} && timeout 60 {BCB_PY} r.py s.py t.py"],
                               capture_output=True, text=True, timeout=90, env={**os.environ, "MPLBACKEND": "Agg"})
            m = re.search(r"@@RES@@(.*)", p.stdout)
            res = json.loads(m.group(1)) if m else {"status": "timeout" if p.returncode == 124 else "crash", "passed": 0}
        except subprocess.TimeoutExpired:
            res = {"status": "timeout", "passed": 0}
    if not res.get("total"):
        res["total"] = nt
    return res


def harness():
    """Run every canonical solution once; tasks that fail here are dropped everywhere."""
    from concurrent.futures import ThreadPoolExecutor
    if os.path.exists(path("harness.jsonl")):
        return
    tasks = load_tasks()
    with ThreadPoolExecutor(48) as ex:
        rs = list(ex.map(lambda t: {"task_id": t["task_id"], **run_tests(t["code_prompt"] + t["canonical_solution"], t["test"])}, tasks))
    write_jsonl(path("harness.jsonl"), rs)


# ---------------------------------------------------------------- stats
def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()


def boot(x, n=4000, seed=0):
    """mean and 95% percentile bootstrap CI."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def fmt(x, d=3):
    m, lo, hi = boot(x)
    return f"{m:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


# ---------------------------------------------------------------- J-lens
def load_jlens(device="cuda"):
    """Pre-fitted Qwen3-14B J-lens: {layer: J_l} with lens(h) = W_U norm(J_l h)."""
    import torch
    from huggingface_hub import hf_hub_download
    ck = torch.load(hf_hub_download(LENS_REPO, LENS_FILE), map_location="cpu", weights_only=True)
    J = ck["J"] if "J" in ck else {l: v / ck["n_done"] for l, v in ck["jacobian_sum"].items()}
    J = {int(l): v.to(torch.bfloat16).to(device) for l, v in J.items() if int(l) in BAND}
    assert set(BAND) <= set(J)
    return J


def jlens_topk(model, J, ids, k=8, chunk=1024):
    """Teacher-force `ids` once; at every position return the top-k tokens of the band-mean J-lens log-probs."""
    import torch
    with torch.no_grad():
        x = torch.tensor(ids)[None].to(model.device)
        hs = model(x, output_hidden_states=True).hidden_states  # hs[l+1] = output of block l
        norm, head, T = model.model.norm, model.lm_head, x.shape[1]
        ti, tv = np.zeros((T, k), np.int32), np.zeros((T, k), np.float16)
        for a in range(0, T, chunk):
            b = min(T, a + chunk)
            acc = sum(torch.log_softmax(head(norm(hs[l + 1][0, a:b] @ J[l].T)).float(), -1) for l in BAND) / len(BAND)
            v, i = acc.topk(k, -1)
            ti[a:b], tv[a:b] = i.cpu().numpy(), v.cpu().numpy()
    return ti, tv
