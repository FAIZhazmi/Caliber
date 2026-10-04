"""Temporary username/password gate for the dashboard (competition period).

Credentials come from AUTH_USERS (environment, .env, or Streamlit secrets) as
``username:password`` pairs separated by ``;``. With no credentials configured the
gate stays closed, so a missing secret can never expose the dashboard.
"""

from __future__ import annotations

import hmac
import time

import streamlit as st

from dashboard.clickup import clickup_setting

MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 60


def parse_users(raw: str) -> dict[str, str]:
    """Parse ``user:pass;user2:pass2``; the first ':' splits user from password."""
    users: dict[str, str] = {}
    for pair in raw.split(";"):
        username, separator, password = pair.partition(":")
        username = username.strip()
        if separator and username and password:
            users[username] = password
    return users


def check_credentials(users: dict[str, str], username: str, password: str) -> bool:
    """Compare in constant time, and always against some password, to avoid leaking which usernames exist."""
    expected = users.get(username.strip(), "")
    matched = hmac.compare_digest(expected.encode(), password.encode())
    return matched and bool(expected)


def _render_login(users: dict[str, str]) -> None:
    st.markdown(
        """
        <style>
          .stApp { background: linear-gradient(180deg, #eef4fc 0%, #e4edfa 100%); }
          header[data-testid="stHeader"], section[data-testid="stSidebar"] { display: none !important; }
          .login-title { color: #0f3d91; font-size: 1.7rem; font-weight: 800; margin: 0; }
          .login-sub { color: #5d7199; margin: 0 0 .5rem 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    _, middle, _ = st.columns([1, 1.1, 1])
    with middle:
        st.write("")
        st.write("")
        with st.container(border=True):
            st.markdown('<p class="login-title">CORE</p>', unsafe_allow_html=True)
            st.markdown(
                '<p class="login-sub">Centralized Operations &amp; Reliability Engine</p>',
                unsafe_allow_html=True,
            )
            if not users:
                st.error("Login is not configured. Set AUTH_USERS in the app secrets.")
                return
            locked_until = st.session_state.get("auth_locked_until", 0.0)
            remaining = int(locked_until - time.time())
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign in", use_container_width=True)
            if remaining > 0:
                st.error(f"Too many failed attempts. Try again in {remaining} seconds.")
            elif submitted:
                if check_credentials(users, username, password):
                    st.session_state["auth_user"] = username.strip()
                    st.session_state["auth_failures"] = 0
                    st.rerun()
                failures = st.session_state.get("auth_failures", 0) + 1
                st.session_state["auth_failures"] = failures
                if failures >= MAX_ATTEMPTS:
                    st.session_state["auth_failures"] = 0
                    st.session_state["auth_locked_until"] = time.time() + LOCKOUT_SECONDS
                    st.error(f"Too many failed attempts. Try again in {LOCKOUT_SECONDS} seconds.")
                else:
                    st.error("Incorrect username or password.")


def require_login() -> str:
    """Block the page with a login form until the visitor signs in; return the username."""
    user = st.session_state.get("auth_user")
    if user:
        return user
    _render_login(parse_users(clickup_setting("AUTH_USERS")))
    st.stop()


def render_logout() -> None:
    """Small sign-out control, shown only to a signed-in visitor."""
    user = st.session_state.get("auth_user")
    if not user:
        return
    columns = st.columns([6, 1])
    columns[0].caption(f"Signed in as {user}")
    if columns[1].button("Sign out", key="auth_signout"):
        st.session_state.pop("auth_user", None)
        st.rerun()
