"""Login portal, hashed credentials and role-based access.

Users live in the same SQLite file as the crowd data (table `users`).
Passwords are stored as salted PBKDF2-HMAC-SHA256 hashes, never in plain text.

Roles
-----
admin  : everything, plus the "User Management" page and destructive actions
viewer : read-only monitoring (can request an account from the login page)

Environment variables (use these when you deploy):
  CAMPUS_ADMIN_USER      default "admin"
  CAMPUS_ADMIN_PASSWORD  default "admin123"   <- CHANGE THIS in production
  SHOW_DEMO_CREDS        "1" shows the demo accounts on the login page, "0" hides them
"""
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time

import pandas as pd
import streamlit as st

import style
from database import DB_DIR, DB_PATH

ITER = 200_000
MAX_TRIES, LOCK_SECONDS = 5, 60
ROLES = ["viewer", "admin"]


def _conn():
    DB_DIR.mkdir(exist_ok=True)
    return sqlite3.connect(DB_PATH)


def _hash(password, salt=None):
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), ITER)
    return f"{salt}${dk.hex()}"


def _verify(password, stored):
    try:
        salt, _ = stored.split("$")
    except ValueError:
        return False
    return hmac.compare_digest(_hash(password, salt), stored)


# ------------------------------------------------------------------ storage
def init_users():
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                full_name TEXT,
                pw TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'viewer',
                active INTEGER NOT NULL DEFAULT 1,
                created TEXT,
                last_login TEXT)""")
        if c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            admin = os.getenv("CAMPUS_ADMIN_USER", "admin").lower()
            c.execute("INSERT INTO users VALUES (?,?,?,?,1,?,NULL)",
                      (admin, "Campus Administrator", _hash(os.getenv("CAMPUS_ADMIN_PASSWORD", "admin123")),
                       "admin", _now()))
            c.execute("INSERT INTO users VALUES (?,?,?,?,1,?,NULL)",
                      ("staff", "Security Staff", _hash("staff123"), "viewer", _now()))


def _now():
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")


def create_user(username, password, full_name="", role="viewer"):
    username = username.strip().lower()
    if not re.fullmatch(r"[a-z0-9_.-]{3,24}", username):
        return False, "Username must be 3-24 characters: letters, numbers, . _ -"
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if role not in ROLES:
        role = "viewer"
    try:
        with _conn() as c:
            c.execute("INSERT INTO users VALUES (?,?,?,?,1,?,NULL)",
                      (username, full_name.strip() or username, _hash(password), role, _now()))
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    return True, "Account created."


def list_users():
    with _conn() as c:
        return pd.read_sql_query(
            "SELECT username, full_name, role, active, created, last_login FROM users ORDER BY role, username", c)


def set_password(username, password):
    with _conn() as c:
        c.execute("UPDATE users SET pw=? WHERE username=?", (_hash(password), username))


def set_active(username, active):
    with _conn() as c:
        c.execute("UPDATE users SET active=? WHERE username=?", (int(active), username))


def set_role(username, role):
    with _conn() as c:
        c.execute("UPDATE users SET role=? WHERE username=?", (role, username))


def delete_user(username):
    with _conn() as c:
        c.execute("DELETE FROM users WHERE username=?", (username,))


def authenticate(username, password):
    username = username.strip().lower()
    with _conn() as c:
        row = c.execute("SELECT username, full_name, pw, role, active FROM users WHERE username=?",
                        (username,)).fetchone()
        if row and row[4] and _verify(password, row[2]):
            c.execute("UPDATE users SET last_login=? WHERE username=?", (_now(), username))
            return {"username": row[0], "name": row[1] or row[0], "role": row[3]}
    return None


# ------------------------------------------------------------------ session
def current_user():
    return st.session_state.get("user")


def is_admin():
    u = current_user()
    return bool(u and u["role"] == "admin")


def logout():
    for k in ("user", "last"):
        st.session_state.pop(k, None)
    st.rerun()


def require_login():
    """Show the login portal and stop the script until someone is signed in."""
    ss = st.session_state
    if ss.get("user"):
        return ss["user"]

    style.login_css()
    ss.setdefault("fails", 0)
    ss.setdefault("locked_until", 0.0)

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown(style.login_brand_html(), unsafe_allow_html=True)
    with right:
        st.markdown("<div class='login-spacer'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<div class='login-title'>Welcome back</div>"
                        "<div class='login-sub'>Sign in to monitor campus crowd levels.</div>",
                        unsafe_allow_html=True)
            t_in, t_up = st.tabs(["Sign in", "Request access"])
            with t_in:
                with st.form("login_form", clear_on_submit=False):
                    u = st.text_input("Username", placeholder="e.g. admin")
                    p = st.text_input("Password", type="password", placeholder="Your password")
                    go = st.form_submit_button("Sign in", type="primary", use_container_width=True)
                if go:
                    wait = ss.locked_until - time.time()
                    if wait > 0:
                        st.error(f"Too many attempts. Try again in {int(wait) + 1} seconds.")
                    else:
                        user = authenticate(u, p) if u and p else None
                        if user:
                            ss.user, ss.fails = user, 0
                            st.rerun()
                        else:
                            ss.fails += 1
                            if ss.fails >= MAX_TRIES:
                                ss.locked_until, ss.fails = time.time() + LOCK_SECONDS, 0
                                st.error(f"Too many attempts. Locked for {LOCK_SECONDS} seconds.")
                            else:
                                st.error("Wrong username or password, or the account is disabled.")
            with t_up:
                with st.form("signup_form", clear_on_submit=True):
                    n = st.text_input("Full name")
                    u2 = st.text_input("Choose a username")
                    p1 = st.text_input("Choose a password", type="password")
                    p2 = st.text_input("Repeat password", type="password")
                    go2 = st.form_submit_button("Create account", use_container_width=True)
                if go2:
                    if p1 != p2:
                        st.error("The two passwords don't match.")
                    else:
                        ok, msg = create_user(u2, p1, n, "viewer")
                        (st.success if ok else st.error)(
                            msg + (" You can sign in now (viewer access)." if ok else ""))
            if os.getenv("SHOW_DEMO_CREDS", "1") == "1":
                st.caption("Demo accounts: admin / admin123 (full access) · staff / staff123 (view only)")
    st.stop()


# ------------------------------------------------------------------ admin page
def user_management_page():
    style.hero("User management", "Create accounts, change roles and reset passwords")
    me = current_user()["username"]
    df = list_users()
    st.dataframe(df, hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.subheader("Add a user")
            with st.form("add_user", clear_on_submit=True):
                n = st.text_input("Full name")
                u = st.text_input("Username")
                p = st.text_input("Temporary password", type="password")
                r = st.selectbox("Role", ROLES)
                if st.form_submit_button("Create user", type="primary"):
                    ok, msg = create_user(u, p, n, r)
                    (st.success if ok else st.error)(msg)
                    if ok:
                        st.rerun()
    with c2:
        with st.container(border=True):
            st.subheader("Manage an existing user")
            target = st.selectbox("User", df["username"].tolist())
            row = df[df.username == target].iloc[0]
            np_ = st.text_input("New password", type="password", key="np")
            a, b, c = st.columns(3)
            if a.button("Reset password"):
                if len(np_) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    set_password(target, np_)
                    st.success("Password updated.")
            if target == me:
                st.caption("You can't disable, demote or delete your own account.")
            else:
                if b.button("Disable" if row.active else "Enable"):
                    set_active(target, not row.active)
                    st.rerun()
                if c.button("Delete"):
                    delete_user(target)
                    st.rerun()
                new_role = st.selectbox("Role", ROLES, index=ROLES.index(row.role), key="role_sel")
                if new_role != row.role:
                    set_role(target, new_role)
                    st.rerun()
