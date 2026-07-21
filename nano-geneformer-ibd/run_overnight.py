"""Overnight batch: prep full 365k atlas, then train + probe two variants
(scale baseline, scale + weight tying). Each stage runs as a subprocess with a
timeout so one hang can't consume the night; all results land under
results/overnight/. Writes summary.json + REPORT.md at the end (and after each
stage, so partial results are always readable).

Run:  python run_overnight.py            (full)
      python run_overnight.py --smoke     (tiny end-to-end test)
"""
import argparse, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(cmd, log_path, timeout):
    """Run cmd, tee to log_path, return (ok, note)."""
    with open(log_path, "w") as f:
        f.write(f"$ {' '.join(cmd)}\n\n"); f.flush()
        try:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                           timeout=timeout, cwd=ROOT, check=True)
            return True, "ok"
        except subprocess.TimeoutExpired:
            return False, f"TIMEOUT after {timeout}s"
        except subprocess.CalledProcessError as e:
            return False, f"exit {e.returncode}"


def write_report(outdir, results, args):
    lines = ["# nano-Geneformer overnight run\n",
             f"data: `{args.data}`  |  epochs: {args.epochs}  |  max_len: {args.max_len}\n",
             "\n## Probe results (held-out patients)\n",
             "| variant | loss | disease embed | disease PCA | celltype embed | celltype PCA |",
             "|---|---|---|---|---|---|"]
    for name, r in results.items():
        if name.startswith("_"):
            continue
        m = r.get("metrics")
        if not m:
            lines.append(f"| {name} | — | {r.get('status','?')} | | | |"); continue
        d, c = m["disease"], m["celltype"]
        lines.append(
            f"| {name} | {m['final_loss']:.3f} "
            f"| {d['embed']['macro_f1']:.3f} | {d['pca']['macro_f1']:.3f} "
            f"| {c['embed']['acc']:.3f} | {c['pca']['acc']:.3f} |")
    lines.append("\n(disease = 3-class macro-F1; celltype = accuracy over "
                 f"{results.get('_nct','?')} classes)\n")
    for name, r in results.items():
        if name.startswith("_"):
            continue
        m = r.get("metrics")
        if m and m.get("col1a1_neighbors"):
            nb = ", ".join(f"{g}({s})" for g, s in m["col1a1_neighbors"][:6])
            lines.append(f"- **{name}** COL1A1 neighbors: {nb}")
    open(os.path.join(outdir, "REPORT.md"), "w").write("\n".join(lines))
    json.dump({k: v for k, v in results.items() if not k.startswith("_")},
              open(os.path.join(outdir, "summary.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_365k")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        a.data, a.epochs = "data_smoke", 1

    outdir = os.path.join(ROOT, "results", "overnight_smoke" if a.smoke else "overnight")
    os.makedirs(outdir, exist_ok=True)
    results = {}
    t_start = time.time()

    def status(msg):
        print(f"[{(time.time()-t_start)/60:6.1f}m] {msg}", flush=True)
        open(os.path.join(outdir, "STATUS.txt"), "a").write(
            f"[{(time.time()-t_start)/60:6.1f}m] {msg}\n")

    # ---- stage 0: prep -------------------------------------------------------
    if not os.path.exists(os.path.join(ROOT, a.data, "tokens.npy")):
        status(f"PREP -> {a.data} (max_len={a.max_len}) ...")
        cmd = [PY, "prepare.py", "--data", "smillie", "--out", a.data,
               "--max-len", str(a.max_len)]
        if a.smoke:
            cmd += ["--subsample", "3000"]
        else:
            cmd += ["--subsample", "0"]
        ok, note = run(cmd, os.path.join(outdir, "prep.log"), 7200)
        status(f"PREP {'done' if ok else 'FAILED: ' + note}")
        if not ok:
            status("prep failed -> abort"); return
    else:
        status(f"PREP skipped ({a.data} exists)")

    # ---- stages: variants ----------------------------------------------------
    variants = [("scale-365k", []), ("scale-365k-tied", ["--tie"])]
    if a.smoke:
        variants = [("smoke-base", []), ("smoke-tied", ["--tie"])]

    for name, extra in variants:
        sdir = os.path.join(outdir, name); os.makedirs(sdir, exist_ok=True)
        status(f"TRAIN {name} ...")
        tcmd = [PY, "train_stage.py", "--data", a.data, "--out", sdir,
                "--epochs", str(a.epochs), "--max-len", str(a.max_len)] + extra
        ok, note = run(tcmd, os.path.join(sdir, "train.log"), 5 * 3600)
        if not ok and not os.path.exists(os.path.join(sdir, "model.pt")):
            status(f"TRAIN {name} FAILED: {note} (no checkpoint) -> skip probe")
            results[name] = {"status": f"train failed: {note}"}; write_report(outdir, results, a); continue
        status(f"TRAIN {name} {'done' if ok else 'partial: ' + note}; PROBE ...")
        pcmd = [PY, "probe_stage.py", "--data", a.data,
                "--ckpt", os.path.join(sdir, "model.pt"), "--out", sdir]
        okp, notep = run(pcmd, os.path.join(sdir, "probe.log"), 40 * 60)
        mpath = os.path.join(sdir, "metrics.json")
        if okp and os.path.exists(mpath):
            m = json.load(open(mpath))
            results[name] = {"status": "ok" if ok else "train-partial", "metrics": m}
            results["_nct"] = m["celltype"]["n_classes"]
            status(f"PROBE {name} done: disease embed={m['disease']['embed']['macro_f1']:.3f} "
                   f"pca={m['disease']['pca']['macro_f1']:.3f} | "
                   f"celltype embed_acc={m['celltype']['embed']['acc']:.3f}")
        else:
            results[name] = {"status": f"probe failed: {notep}"}
            status(f"PROBE {name} FAILED: {notep}")
        write_report(outdir, results, a)

    status(f"ALL DONE in {(time.time()-t_start)/60:.1f}m -> {outdir}/REPORT.md")


if __name__ == "__main__":
    main()
