import os

# Settings refuses the placeholder CONTACT_URL. Tests must not depend on the developer's .env for it,
# so give them a reserved-domain contact (environment variables override .env).
os.environ.setdefault("CONTACT_URL", "https://contact.test/research-assistant")
