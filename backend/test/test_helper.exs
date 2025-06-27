ExUnit.start()

# Set up Mox for mocking
Mox.defmock(Sup.SpamDetection.MockClient, for: Sup.SpamDetection.ClientBehaviour)

# Configure test environment
Application.put_env(:logger, :level, :warn)

# Start Bypass for HTTP mocking
Application.ensure_all_started(:bypass)

# Import test helpers
Code.require_file("support/factory.ex", __DIR__)

# Set up ExMachina
{:ok, _} = Application.ensure_all_started(:ex_machina)
