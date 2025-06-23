defmodule Sup.Messaging.KeyBundle do
  @moduledoc """
  Key bundle structure for Signal Protocol key exchange.
  """

  defstruct [
    :user_id,
    :identity_key,
    :signed_prekey_id,
    :signed_prekey,
    :signed_prekey_signature,
    :prekey_id,
    :prekey
  ]
end
