import os
from app import create_app, db
from app.auth import User
from werkzeug.security import generate_password_hash

# Set an environment variable to prevent the app from running in debug mode for this script
os.environ['FLASK_DEBUG'] = '0'

app = create_app()

with app.app_context():
    # --- Configuration for your admin user ---
    ADMIN_EMAIL = "Rajagrawal350@gmail.com"  # Change this to your desired admin email
    ADMIN_PASSWORD = "Radharanikijai"   # Change this to a strong password

    # Check if the admin user already exists
    if User.query.filter_by(email=ADMIN_EMAIL).first():
        print(f"Admin user with email {ADMIN_EMAIL} already exists.")
    else:
        # Create the new admin user
        hashed_password = generate_password_hash(ADMIN_PASSWORD)
        admin_user = User(
            email=ADMIN_EMAIL,
            password=hashed_password,
            is_admin=True  # Set this user as an admin
        )
        db.session.add(admin_user)
        db.session.commit()
        print(f"Successfully created admin user: {ADMIN_EMAIL}")