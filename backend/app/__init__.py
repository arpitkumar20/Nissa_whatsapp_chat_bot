from flask import Flask
from .core.config import settings
from .core.logging import configure_logging

def create_app():
    app = Flask(__name__)
    # app.config['SECRET_KEY'] = settings.SECRET_KEY
    configure_logging(app)

    # register blueprints
    from .api.routes_whatsapp import bp as whatsapp_bp
    from .api.routes_chat import bp as chat_bp
    from .api.routes_admin import bp as admin_bp
    from .api.routes_booking import bp as booking_bp

    app.register_blueprint(whatsapp_bp, url_prefix="/webhooks/whatsapp")
    app.register_blueprint(chat_bp, url_prefix="/api/chat")
    app.register_blueprint(admin_bp, url_prefix="/api/admin")
    app.register_blueprint(booking_bp, url_prefix="/api/booking")

    return app
