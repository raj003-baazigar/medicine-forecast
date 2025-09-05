from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)

from app import db, create_app
from app.models import User  # or wherever your User model is defined

app = create_app()

with app.app_context():
    db.create_all()
    
    # Optional: Add a test user
    if not User.query.filter_by(email="test@example.com").first():
        new_user = User(email="test@example.com", password="hashed_password")
        db.session.add(new_user)
        db.session.commit()

    print(" Database created with a sample user.")
