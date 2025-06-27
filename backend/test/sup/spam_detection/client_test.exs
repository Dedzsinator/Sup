defmodule Sup.SpamDetection.ClientTest do
  use ExUnit.Case
  import Mox

  alias Sup.SpamDetection.Client

  setup :verify_on_exit!

  setup do
    bypass = Bypass.open()
    Application.put_env(:sup, :spam_detection_url, "http://localhost:#{bypass.port}")
    Application.put_env(:sup, :spam_detection_api_key, "test-api-key")
    {:ok, bypass: bypass}
  end

  describe "check_spam/3" do
    test "returns spam result when API responds successfully", %{bypass: bypass} do
      Bypass.expect_once(bypass, "POST", "/predict", fn conn ->
        {:ok, body, conn} = Plug.Conn.read_body(conn)
        assert Jason.decode!(body)["text"] == "Buy cheap viagra now!"

        response = %{
          "is_spam" => true,
          "confidence" => 0.95,
          "model_type" => "neural_network",
          "processing_time_ms" => 25.3
        }

        conn
        |> Plug.Conn.put_resp_content_type("application/json")
        |> Plug.Conn.resp(200, Jason.encode!(response))
      end)

      assert {:ok, result} = Client.check_spam("Buy cheap viagra now!", "user123")
      assert result["is_spam"] == true
      assert result["confidence"] == 0.95
      assert result["model_type"] == "neural_network"
    end

    test "returns error when API returns non-200 status", %{bypass: bypass} do
      Bypass.expect_once(bypass, "POST", "/predict", fn conn ->
        Plug.Conn.resp(conn, 500, "Internal Server Error")
      end)

      assert {:error, :api_error} = Client.check_spam("test message", "user123")
    end

    test "returns error when API response is invalid JSON", %{bypass: bypass} do
      Bypass.expect_once(bypass, "POST", "/predict", fn conn ->
        conn
        |> Plug.Conn.put_resp_content_type("application/json")
        |> Plug.Conn.resp(200, "invalid json")
      end)

      assert {:error, :decode_error} = Client.check_spam("test message", "user123")
    end

    test "returns timeout error when request times out", %{bypass: bypass} do
      Bypass.expect_once(bypass, "POST", "/predict", fn conn ->
        # Simulate slow response
        Process.sleep(6000)
        Plug.Conn.resp(conn, 200, Jason.encode!(%{}))
      end)

      assert {:error, :timeout} = Client.check_spam("test message", "user123")
    end

    test "uses anonymous user_id when nil is provided", %{bypass: bypass} do
      Bypass.expect_once(bypass, "POST", "/predict", fn conn ->
        {:ok, body, conn} = Plug.Conn.read_body(conn)
        assert Jason.decode!(body)["user_id"] == "anonymous"

        response = %{"is_spam" => false, "confidence" => 0.1}

        conn
        |> Plug.Conn.put_resp_content_type("application/json")
        |> Plug.Conn.resp(200, Jason.encode!(response))
      end)

      assert {:ok, _result} = Client.check_spam("test message", nil)
    end
  end

  describe "check_spam_batch/1" do
    test "processes batch messages successfully", %{bypass: bypass} do
      Bypass.expect_once(bypass, "POST", "/predict/batch", fn conn ->
        {:ok, body, conn} = Plug.Conn.read_body(conn)
        decoded_body = Jason.decode!(body)
        assert length(decoded_body["messages"]) == 2

        response = %{
          "predictions" => [
            %{"is_spam" => true, "confidence" => 0.9},
            %{"is_spam" => false, "confidence" => 0.1}
          ]
        }

        conn
        |> Plug.Conn.put_resp_content_type("application/json")
        |> Plug.Conn.resp(200, Jason.encode!(response))
      end)

      messages = [
        %{message: "Buy now!", user_id: "user1"},
        %{text: "Hello world", user_id: "user2"}
      ]

      assert {:ok, predictions} = Client.check_spam_batch(messages)
      assert length(predictions) == 2
      assert Enum.at(predictions, 0)["is_spam"] == true
      assert Enum.at(predictions, 1)["is_spam"] == false
    end

    test "handles missing message fields gracefully", %{bypass: bypass} do
      Bypass.expect_once(bypass, "POST", "/predict/batch", fn conn ->
        {:ok, body, conn} = Plug.Conn.read_body(conn)
        decoded_body = Jason.decode!(body)
        messages = decoded_body["messages"]

        # Check that missing fields are handled
        assert Enum.at(messages, 0)["user_id"] == "anonymous"
        assert Enum.at(messages, 0)["text"] == nil

        response = %{"predictions" => [%{"is_spam" => false, "confidence" => 0.1}]}

        conn
        |> Plug.Conn.put_resp_content_type("application/json")
        |> Plug.Conn.resp(200, Jason.encode!(response))
      end)

      # Empty message data
      messages = [%{}]
      assert {:ok, _predictions} = Client.check_spam_batch(messages)
    end
  end

  describe "health_check/0" do
    test "returns true when service is healthy", %{bypass: bypass} do
      Bypass.expect_once(bypass, "GET", "/health", fn conn ->
        Plug.Conn.resp(conn, 200, "OK")
      end)

      assert Client.health_check() == true
    end

    test "returns false when service is unhealthy", %{bypass: bypass} do
      Bypass.expect_once(bypass, "GET", "/health", fn conn ->
        Plug.Conn.resp(conn, 500, "Error")
      end)

      assert Client.health_check() == false
    end
  end

  describe "get_fallback_result/1" do
    test "detects spam patterns correctly" do
      spam_message = "WIN MONEY NOW! Visit our website for free viagra!"
      result = Client.get_fallback_result(spam_message)

      assert result["is_spam"] == true
      assert result["confidence"] > 0
      assert result["model_type"] == "fallback_rule_based"
      assert result["metadata"]["service_status"] == "unavailable"
    end

    test "detects excessive capitals as spam" do
      caps_message = "THIS IS ALL CAPS MESSAGE!!!"
      result = Client.get_fallback_result(caps_message)

      assert result["is_spam"] == true
      assert result["model_type"] == "fallback_rule_based"
    end

    test "classifies normal messages as not spam" do
      normal_message = "Hello, how are you doing today?"
      result = Client.get_fallback_result(normal_message)

      assert result["is_spam"] == false
      assert result["confidence"] == 0.0
    end
  end

  describe "check_spam_with_fallback/3" do
    test "uses API result when service is available", %{bypass: bypass} do
      Bypass.expect_once(bypass, "POST", "/predict", fn conn ->
        response = %{"is_spam" => true, "confidence" => 0.95}

        conn
        |> Plug.Conn.put_resp_content_type("application/json")
        |> Plug.Conn.resp(200, Jason.encode!(response))
      end)

      result = Client.check_spam_with_fallback("spam message", "user123")
      assert result["is_spam"] == true
      assert result["confidence"] == 0.95
    end

    test "falls back to rule-based detection when service fails", %{bypass: bypass} do
      Bypass.down(bypass)

      spam_message = "WIN MONEY NOW!"
      result = Client.check_spam_with_fallback(spam_message, "user123")

      assert result["is_spam"] == true
      assert result["model_type"] == "fallback_rule_based"
      assert result["metadata"]["service_status"] == "unavailable"
    end
  end
end
