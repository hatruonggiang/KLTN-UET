import os
import sys
import json
import uuid
from flask import Flask, render_template, request, send_file, redirect, url_for

# Thêm thư mục core vào path để các file bên trong import lẫn nhau được
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, "core"))

from core.models import Tournament
from core.ga import run_ga
from core.nsga2 import run_nsga2, pick_representative

# Cấu hình để Flask tìm kiếm template ở thư mục hiện tại thay vì thư mục /templates mặc định
app = Flask(__name__, template_folder='.')
app.config['UPLOAD_FOLDER'] = os.path.join(base_dir, 'uploads')
app.config['RESULT_FOLDER'] = os.path.join(base_dir, 'results')

# Đảm bảo các thư mục tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    template_dir = os.path.join(base_dir, 'data')
    templates = [f for f in os.listdir(template_dir) if f.endswith('.json')]
    return render_template('index.html', templates=templates)

@app.route('/run', methods=['POST'])
def run_algorithm():
    if 'file' not in request.files:
        return "Không tìm thấy file upload", 400
    
    file = request.files['file']
    algo = request.form.get('algorithm')
    
    if file.filename == '':
        return "Chưa chọn file", 400

    # 1. Lưu file data tạm thời
    input_filename = f"data_{uuid.uuid4()}.json"
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
    file.save(input_path)

    try:
        # 2. Khởi tạo Tournament
        tournament = Tournament(input_path)
        
        # 3. Chạy thuật toán và thu thập kết quả
        results = []

        def prepare_result(ind, label):
            if not ind: return None
            out_fn = f"schedule_{algo}_{uuid.uuid4().hex[:8]}.json"
            out_path = os.path.join(app.config['RESULT_FOLDER'], out_fn)
            
            obj_str = ""
            if hasattr(ind, 'objectives') and ind.objectives:
                obj_str = f"F1={ind.objectives[0]:.0f}, F2={ind.objectives[1]:.0f}, F3={ind.objectives[2]:.0f}"
            
            data = {
                "tournament_name": tournament.name,
                "algorithm": algo.upper(),
                "label": label,
                "total_matches": len(ind.matches),
                "matches": [
                    {
                        "round": m.round,
                        "timeslot_id": m.timeslot_id,
                        "home_team": tournament.get_team(m.home_team_id).name,
                        "away_team": tournament.get_team(m.away_team_id).name,
                        "stadium": tournament.get_team(m.home_team_id).stadium,
                        "time": tournament.get_match_datetime(m.round, m.timeslot_id).strftime("%Y-%m-%d %H:%M")
                    }
                    for m in sorted(ind.matches, key=lambda x: (x.round, x.timeslot_id))
                ]
            }
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return {
                "filename": out_fn,
                "label": label,
                "fitness": f"{ind.fitness:.1f}" if ind.fitness else "0",
                "objectives": obj_str,
                "num_matches": len(ind.matches),
                "num_teams": tournament.num_teams
            }

        if algo == 'ga':
            print("[WEB] Đang chạy thuật toán GA...")
            best_ind, _ = run_ga(tournament)
            res = prepare_result(best_ind, "Lịch tối ưu (GA)")
            if res: results.append(res)
        else:
            print("[WEB] Đang chạy thuật toán NSGA-II...")
            pareto, _ = run_nsga2(tournament)
            # Lấy 3 lịch đại diện cho 3 mục tiêu
            configs = [
                ("fairness", "Ưu tiên Công bằng (F1)"),
                ("quality", "Ưu tiên Chất lượng (F2)"),
                ("broadcast", "Ưu tiên Phát sóng (F3)")
            ]
            for strat, label in configs:
                ind = pick_representative(pareto, strategy=strat)
                res = prepare_result(ind, label)
                if res: results.append(res)

        return render_template('result.html', results=results, algo=algo.upper())

    except Exception as e:
        return f"Đã xảy ra lỗi: {str(e)}", 500

@app.route('/schedule/<filename>')
def view_schedule(filename):
    # Trình diễn giao diện xem lịch chi tiết cho một file kết quả cụ thể
    return render_template('schedule.html', filename=filename)

@app.route('/download/template/<filename>')
def download_template(filename):
    # File mẫu nằm trong thư mục data ở thư mục gốc
    template_path = os.path.join(base_dir, 'data', filename)
    return send_file(template_path, as_attachment=True)

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)