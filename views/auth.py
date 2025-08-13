from flask import Blueprint, render_template, request, redirect, url_for, session

auth_bp = Blueprint('auth', __name__)

def authenticate(username, password):
    """Simple authentication function (extend to database if needed)."""
    return username == "admin" and password == "password"

@auth_bp.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if authenticate(username, password):
            session['logged_in'] = True
            return redirect(url_for('dashboard.dashboard'))
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@auth_bp.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('auth.login'))