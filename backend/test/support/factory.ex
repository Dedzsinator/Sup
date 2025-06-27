defmodule Sup.Factory do
  @moduledoc """
  Test data factory for generating test fixtures
  """

  use ExMachina

  def spam_message_factory do
    %{
      text: sequence("spam_message", &"Buy cheap products now #{&1}!"),
      user_id: sequence("user", &"user#{&1}"),
      metadata: %{
        timestamp: DateTime.utc_now() |> DateTime.to_iso8601()
      }
    }
  end

  def ham_message_factory do
    %{
      text: sequence("ham_message", &"Hello, how are you today #{&1}?"),
      user_id: sequence("user", &"user#{&1}"),
      metadata: %{
        timestamp: DateTime.utc_now() |> DateTime.to_iso8601()
      }
    }
  end

  def spam_detection_response_factory do
    %{
      "is_spam" => true,
      "confidence" => 0.95,
      "model_type" => "neural_network",
      "processing_time_ms" => 25.3,
      "metadata" => %{
        "score" => 0.95,
        "service_status" => "available"
      }
    }
  end

  def ham_detection_response_factory do
    %{
      "is_spam" => false,
      "confidence" => 0.05,
      "model_type" => "neural_network",
      "processing_time_ms" => 23.1,
      "metadata" => %{
        "score" => 0.05,
        "service_status" => "available"
      }
    }
  end

  def batch_prediction_response_factory do
    %{
      "predictions" => [
        build(:spam_detection_response),
        build(:ham_detection_response)
      ],
      "total_processed" => 2,
      "processing_time_ms" => 45.2
    }
  end
end
