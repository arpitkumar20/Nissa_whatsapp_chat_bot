from .routes_chat import chat_bp
from .routes_admin import admin_bp

def init_routes(app):
    app.register_blueprint(chat_bp, url_prefix="/chat")
    app.register_blueprint(admin_bp, url_prefix="/admin")