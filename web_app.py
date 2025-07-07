from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import os
import subprocess
from pathlib import Path
import shutil
import sys

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['RESULT_FOLDER'] = 'generated_papers/'
app.config['CONFIG_FILE'] = 'experiment_configs/MATH_agentlab.yaml'  # Your existing config

# Ensure folders exist
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'POST':
        # Run the paper generation with the fixed config file
        try:
            # Get path to virtual environment's Python
            # venv_python = os.path.join('venv_agent_lab', 'Scripts', 'python')  # Windows
            # if not os.path.exists(venv_python):  # Try Linux/Mac path if Windows path doesn't exist
            #     venv_python = os.path.join('venv_agent_lab', 'bin', 'python')
            # print(f'Using Python interpreter: {venv_python}')
            result = subprocess.run(
                ['python', 'ai_lab_repo.py', '--yaml-location', app.config['CONFIG_FILE']],
                capture_output=True,
                text=True,
                shell=True  # Needed for Windows
            )
            print(f'Subprocess output: {app.config['CONFIG_FILE']}')
            if result.returncode != 0:
                flash(f'Error generating papers: {result.stderr}')
                return redirect(url_for('index'))
                
            # Find and copy generated papers to results folder
            generated_dir = "MATH_research_dir"  # From your code
            if os.path.exists(generated_dir):
                # Clear previous results
                shutil.rmtree(app.config['RESULT_FOLDER'], ignore_errors=True)
                os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
                
                # Copy all generated files
                for root, _, files in os.walk(generated_dir):
                    for file in files:
                        if file.endswith('.pdf') or file.endswith('.txt'):
                            src_path = os.path.join(root, file)
                            dest_path = os.path.join(app.config['RESULT_FOLDER'], file)
                            shutil.copy2(src_path, dest_path)
                
                return redirect(url_for('results'))
            else:
                flash('Paper generation completed but no output files found')
                return redirect(url_for('index'))
                
        except Exception as e:
            flash(f'Error during paper generation: {str(e)}')
            return redirect(url_for('index'))
    
    return redirect(url_for('index'))

@app.route('/results')
def results():
    files = []
    for filename in os.listdir(app.config['RESULT_FOLDER']):
        if filename.endswith('.pdf') or filename.endswith('.txt'):
            files.append(filename)
    return render_template('results.html', files=files)

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)