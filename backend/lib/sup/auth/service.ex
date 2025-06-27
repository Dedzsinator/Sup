defmodule Sup.Auth.Service do
  @moduledoc """
  Authentication service handling user registration, login, and token management.
  """

  alias Sup.Auth.{User, Guardian}
  alias Sup.Repo

  def register(%{email: email, password: password, username: username}) do
    # Check if user already exists
    case Repo.get_by(User, email: email) do
      nil ->
        user_attrs = %{
          email: email,
          username: username,
          password_hash: Argon2.hash_pwd_salt(password)
        }

        case User.changeset(%User{}, user_attrs) |> Repo.insert() do
          {:ok, user} ->
            {:ok, token, _claims} = Guardian.encode_and_sign(user)
            {:ok, User.public_fields(user), token}

          {:error, changeset} ->
            {:error, extract_errors(changeset)}
        end

      _user ->
        {:error, "user_already_exists"}
    end
  end

  def login(%{email: email, password: password}) do
    case Repo.get_by(User, email: email) do
      nil ->
        # Prevent timing attacks
        Argon2.no_user_verify()
        {:error, "invalid_credentials"}

      user ->
        if Argon2.verify_pass(password, user.password_hash) do
          {:ok, token, _claims} = Guardian.encode_and_sign(user)
          {:ok, User.public_fields(user), token}
        else
          {:error, "invalid_credentials"}
        end
    end
  end

  def verify_token(token) do
    case Guardian.decode_and_verify(token) do
      {:ok, %{"sub" => user_id}} ->
        case Repo.get(User, user_id) do
          nil -> {:error, "user_not_found"}
          user -> {:ok, User.public_fields(user)}
        end

      _ ->
        {:error, "invalid_token"}
    end
  end

  defp extract_errors(changeset) do
    changeset.errors
    |> Enum.map(fn {field, {message, _}} -> "#{field} #{message}" end)
    |> Enum.join(", ")
  end
end
