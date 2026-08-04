from google.auth import default

credentials, project = default()

print("Project:", project)

if hasattr(credentials, "service_account_email"):
    print("Service Account:", credentials.service_account_email)
else:
    print("Credentials Type:", type(credentials))