from supabase import create_client

supabase = create_client(
    "https://mgclekcuskkgfnffvpdj.supabase.co",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1nY2xla2N1c2trZ2ZuZmZ2cGRqIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MjYzNTI4NiwiZXhwIjoyMDg4MjExMjg2fQ.6Mfwv_o11WNkiFkPfcHN9af8bjLPUg4yo-0lExWDDuA"
)

print("Testing Supabase connection...")

# Test users_profile table
print("\n1. Testing users_profile table:")
try:
    result = supabase.table('users_profile').select('*').limit(1).execute()
    print(f"   ✅ Table exists. Rows: {len(result.data)}")
    if result.data:
        print(f"   Columns: {list(result.data[0].keys())}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test stories table
print("\n2. Testing stories table:")
try:
    result = supabase.table('stories').select('*').limit(1).execute()
    print(f"   ✅ Table exists. Rows: {len(result.data)}")
    if result.data:
        print(f"   Columns: {list(result.data[0].keys())}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test insert into users_profile
print("\n3. Testing insert (to check required columns):")
try:
    test_data = {
        "email": "test@example.com",
        "password_hash": "test_hash"
    }
    result = supabase.table('users_profile').insert(test_data).execute()
    print(f"   ❌ This should have failed if columns don't match")
except Exception as e:
    error_msg = str(e)
    if "does not exist" in error_msg:
        print(f"   ℹ️  Column mismatch: {error_msg}")
    else:
        print(f"   Error: {e}")
