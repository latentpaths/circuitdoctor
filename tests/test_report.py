import json

from circuitdoctor.report import report
from circuitdoctor.storage import write_json


def test_report_embeds_evidence_without_executable_data(tmp_path):
    root = tmp_path / "run"
    row = {"id": "example", "text": "</script><script>alert('x')</script>", "sample": 0}
    write_json(root / "evaluation/baseline/00-00000.json", {"rows": [row]})
    write_json(root / "evaluation/edited/00-00000.json", {"rows": [{**row, "text": "repaired"}]})
    out = tmp_path / "report.html"
    report(root, out)
    html = out.read_text()
    assert "</script><script>alert" not in html
    payload = html.split('<script id="run-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    data = json.loads(payload)
    assert data["answer_pairs"][0]["original"] == row["text"]
    assert data["answer_pairs"][0]["edited"] == "repaired"
    assert data["answer_pairs_total"] == 1
