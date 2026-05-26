import os
import sys
import json
import uuid
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, "core"))

from core.models import Tournament
from core.ga import run_ga
from core.nsga2 import run_nsga2, pick_representative

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = os.path.join(base_dir, 'uploads')
app.config['RESULT_FOLDER'] = os.path.join(base_dir, 'results')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return jsonify({"status": "Football Scheduler API is running"})

@app.route('/api/templates')
def get_templates():
    template_dir = os.path.join(base_dir, 'data')
    templates = [f for f in os.listdir(template_dir) if f.endswith('.json')]
    return jsonify(templates)

@app.route('/download/template/<filename>')
def download_template(filename):
    path = os.path.join(base_dir, 'data', filename)
    if not os.path.exists(path):
        return jsonify({"error": "File không tồn tại"}), 404
    return send_file(path, as_attachment=True)

@app.route('/run', methods=['POST'])
def run_algorithm():
    if 'file' not in request.files:
        return jsonify({"error": "Không tìm thấy file"}), 400

    file = request.files['file']
    algo = request.form.get('algorithm', 'ga')

    if file.filename == '':
        return jsonify({"error": "Chưa chọn file"}), 400

    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"data_{uuid.uuid4()}.json")
    file.save(input_path)

    try:
        tournament = Tournament(input_path)
        results = []

        def prepare_result(ind, label):
            if not ind:
                return None
            out_fn  = f"schedule_{uuid.uuid4().hex[:8]}.json"
            out_path = os.path.join(app.config['RESULT_FOLDER'], out_fn)

            obj_str = ""
            if hasattr(ind, 'objectives') and ind.objectives:
                f1, f2, f3 = ind.objectives
                obj_str = f"F1={f1:.0f}, F2={f2:.0f}, F3={f3:.0f}"

            data = {
                "label": label,
                "total_matches": len(ind.matches),
                "matches": [
                    {
                        "round":     m.round,
                        "home_team": tournament.get_team(m.home_team_id).name,
                        "away_team": tournament.get_team(m.away_team_id).name,
                        "stadium":   tournament.get_team(m.home_team_id).stadium,
                        "time":      tournament.get_match_datetime(
                                         m.round, m.timeslot_id
                                     ).strftime("%Y-%m-%d %H:%M")
                    }
                    for m in sorted(ind.matches, key=lambda x: (x.round, x.timeslot_id))
                ]
            }
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            return {
                "filename":    out_fn,
                "label":       label,
                "fitness":     f"{ind.fitness:.1f}" if ind.fitness else "0",
                "objectives":  obj_str,
                "num_matches": len(ind.matches),
                "num_teams":   tournament.num_teams
            }

        if algo == 'ga':
            best_ind, _ = run_ga(tournament)
            res = prepare_result(best_ind, "Lịch tối ưu (GA)")
            if res:
                results.append(res)
        else:
            pareto, _, _duration = run_nsga2(tournament)
            for strat, label in [
                ("fairness",  "Ưu tiên Công bằng (F1)"),
                ("quality",   "Ưu tiên Chất lượng (F2)"),
                ("broadcast", "Ưu tiên Phát sóng (F3)"),
            ]:
                ind = pick_representative(pareto, strategy=strat)
                res = prepare_result(ind, label)
                if res:
                    results.append(res)

        return jsonify({"algo": algo.upper(), "results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/schedule/<filename>')
def get_schedule(filename):
    path = os.path.join(app.config['RESULT_FOLDER'], filename)
    if not os.path.exists(path):
        return jsonify({"error": "File không tồn tại"}), 404
    with open(path, encoding='utf-8') as f:
        return jsonify(json.load(f))

@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(app.config['RESULT_FOLDER'], filename)
    if not os.path.exists(path):
        return jsonify({"error": "File không tồn tại"}), 404
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)