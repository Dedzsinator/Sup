defmodule Sup.Auth.User do
  @moduledoc """
  User schema for authentication and profile management.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  schema "users" do
    field(:email, :string)
    field(:username, :string)
    field(:password_hash, :string)
    field(:phone, :string)
    field(:avatar_url, :string)
    field(:is_online, :boolean, default: false)
    field(:last_seen, :utc_datetime)
    field(:push_token, :string)

    timestamps()
  end

  def changeset(user, attrs) do
    user
    |> cast(attrs, [:email, :username, :password_hash, :phone, :avatar_url])
    |> validate_required([:email, :username, :password_hash])
    |> validate_format(:email, ~r/^[^\s]+@[^\s]+\.[^\s]+$/)
    |> validate_length(:username, min: 2, max: 50)
    |> validate_length(:password_hash, min: 8)
    |> unique_constraint(:email)
    |> unique_constraint(:username)
  end

  def public_fields(user) do
    %{
      id: user.id,
      email: user.email,
      username: user.username,
      avatar_url: user.avatar_url,
      is_online: user.is_online,
      last_seen: user.last_seen
    }
  end
end
