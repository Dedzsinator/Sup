defmodule Sup.Security.RBAC do
  @moduledoc """
  Role-Based Access Control system for the Sup application.
  Manages user roles, permissions, and access control.
  """

  use Ecto.Schema
  alias Sup.Repo
  alias Sup.Auth.User

  # Role definitions
  @roles %{
    "admin" => %{
      name: "Administrator",
      description: "Full system access",
      permissions: [
        "user:*",
        "message:*",
        "channel:*",
        "system:*",
        "audit:*"
      ]
    },
    "moderator" => %{
      name: "Moderator",
      description: "Content moderation and user management",
      permissions: [
        "user:read",
        "user:suspend",
        "user:warn",
        "message:read",
        "message:delete",
        "message:moderate",
        "channel:read",
        "channel:moderate"
      ]
    },
    "user" => %{
      name: "User",
      description: "Standard user access",
      permissions: [
        "message:create",
        "message:read",
        "message:update_own",
        "message:delete_own",
        "channel:read",
        "channel:join",
        "channel:leave",
        "user:read_public",
        "user:update_own"
      ]
    },
    "guest" => %{
      name: "Guest",
      description: "Limited read-only access",
      permissions: [
        "message:read_public",
        "channel:read_public",
        "user:read_public"
      ]
    }
  }

  # Permission categories and actions
  @permissions %{
    "user" => [
      "create",
      "read",
      "read_public",
      "update",
      "update_own",
      "delete",
      "suspend",
      "warn",
      "*"
    ],
    "message" => [
      "create",
      "read",
      "read_public",
      "update",
      "update_own",
      "delete",
      "delete_own",
      "moderate",
      "*"
    ],
    "channel" => [
      "create",
      "read",
      "read_public",
      "update",
      "delete",
      "join",
      "leave",
      "moderate",
      "*"
    ],
    "system" => ["read", "update", "restart", "backup", "*"],
    "audit" => ["read", "export", "*"]
  }

  @doc """
  Get all available roles
  """
  def get_roles, do: @roles

  @doc """
  Get role information
  """
  def get_role(role_name) when is_binary(role_name) do
    Map.get(@roles, role_name)
  end

  @doc """
  Get user's effective permissions
  """
  def get_user_permissions(user) do
    case get_role(user.role || "user") do
      nil -> []
      role -> role.permissions
    end
  end

  @doc """
  Check if user has permission
  """
  def has_permission?(user, resource, action) do
    permissions = get_user_permissions(user)

    # Check for wildcard permissions
    wildcard_permission = "#{resource}:*"
    specific_permission = "#{resource}:#{action}"

    cond do
      "*" in permissions -> true
      wildcard_permission in permissions -> true
      specific_permission in permissions -> true
      check_ownership_permission(user, resource, action) -> true
      true -> false
    end
  end

  @doc """
  Check multiple permissions (user must have ALL)
  """
  def has_all_permissions?(user, permission_checks) do
    Enum.all?(permission_checks, fn {resource, action} ->
      has_permission?(user, resource, action)
    end)
  end

  @doc """
  Check multiple permissions (user must have ANY)
  """
  def has_any_permission?(user, permission_checks) do
    Enum.any?(permission_checks, fn {resource, action} ->
      has_permission?(user, resource, action)
    end)
  end

  @doc """
  Update user role
  """
  def update_user_role(user, new_role) do
    if Map.has_key?(@roles, new_role) do
      changeset = User.changeset(user, %{role: new_role})

      case Repo.update(changeset) do
        {:ok, updated_user} ->
          # Log the role change
          Sup.Security.AuditLog.log_admin_action(
            "role_change",
            user.id,
            %{old_role: user.role, new_role: new_role},
            # This should be the admin user ID
            nil,
            %{}
          )

          {:ok, updated_user}

        {:error, changeset} ->
          {:error, changeset}
      end
    else
      {:error, :invalid_role}
    end
  end

  @doc """
  Check if user can perform action on resource with ownership check
  """
  def can_access_resource?(user, resource_type, _resource_id, action, resource_owner_id \\ nil) do
    # First check basic permissions
    if has_permission?(user, resource_type, action) do
      true
    else
      # Check ownership-based permissions
      case action do
        action when action in ["update_own", "delete_own"] ->
          user.id == resource_owner_id and
            has_permission?(user, resource_type, String.replace(action, "_own", ""))

        _ ->
          false
      end
    end
  end

  @doc """
  Get permission requirements for an endpoint
  """
  def get_endpoint_permissions(method, path) do
    case {method, path} do
      # User management endpoints
      {"GET", "/api/users"} -> [{"user", "read"}]
      {"GET", "/api/users/" <> _} -> [{"user", "read"}]
      {"PUT", "/api/users/" <> _} -> [{"user", "update"}]
      {"DELETE", "/api/users/" <> _} -> [{"user", "delete"}]
      # Message endpoints
      {"GET", "/api/messages"} -> [{"message", "read"}]
      {"POST", "/api/messages"} -> [{"message", "create"}]
      {"PUT", "/api/messages/" <> _} -> [{"message", "update"}]
      {"DELETE", "/api/messages/" <> _} -> [{"message", "delete"}]
      # Channel endpoints
      {"GET", "/api/channels"} -> [{"channel", "read"}]
      {"POST", "/api/channels"} -> [{"channel", "create"}]
      {"PUT", "/api/channels/" <> _} -> [{"channel", "update"}]
      {"DELETE", "/api/channels/" <> _} -> [{"channel", "delete"}]
      # Admin endpoints
      {"GET", "/api/admin/users"} -> [{"user", "*"}]
      {"GET", "/api/admin/audit"} -> [{"audit", "read"}]
      {"GET", "/api/admin/system"} -> [{"system", "read"}]
      # Handle special POST endpoints
      {"POST", path} -> get_post_endpoint_permissions(path)
      # Default - no specific permissions required
      _ -> []
    end
  end

  # Helper function for POST endpoint permissions
  defp get_post_endpoint_permissions(path) do
    cond do
      String.contains?(path, "/api/channels/") and String.ends_with?(path, "/join") ->
        [{"channel", "join"}]

      String.contains?(path, "/api/channels/") and String.ends_with?(path, "/leave") ->
        [{"channel", "leave"}]

      true ->
        []
    end
  end

  @doc """
  Validate role assignment (check if user can assign role)
  """
  def can_assign_role?(assigner, target_role) do
    _assigner_permissions = get_user_permissions(assigner)

    cond do
      # Admins can assign any role
      has_permission?(assigner, "user", "*") -> true
      # Moderators can only assign user and guest roles
      has_permission?(assigner, "user", "update") and target_role in ["user", "guest"] -> true
      # Users cannot assign roles
      true -> false
    end
  end

  @doc """
  Get hierarchical role level (higher number = more permissions)
  """
  def get_role_level(role) do
    case role do
      "admin" -> 4
      "moderator" -> 3
      "user" -> 2
      "guest" -> 1
      _ -> 0
    end
  end

  @doc """
  Check if user can perform action on another user (hierarchy check)
  """
  def can_act_on_user?(actor, target) do
    actor_level = get_role_level(actor.role || "user")
    target_level = get_role_level(target.role || "user")

    # Can only act on users with lower or equal role level
    actor_level >= target_level
  end

  @doc """
  Get available actions for a resource type
  """
  def get_resource_actions(resource_type) do
    Map.get(@permissions, resource_type, [])
  end

  @doc """
  Validate permission format
  """
  def valid_permission?(permission) when is_binary(permission) do
    case String.split(permission, ":", parts: 2) do
      [resource, action] ->
        valid_resource?(resource) and valid_action_for_resource?(resource, action)

      _ ->
        false
    end
  end

  def valid_permission?(_), do: false

  # Private functions

  defp check_ownership_permission(_user, _resource, action) do
    # This would be expanded based on specific business logic
    # For now, users can always perform actions on their own resources
    case action do
      "update_own" -> true
      "delete_own" -> true
      "read_own" -> true
      _ -> false
    end
  end

  defp valid_resource?(resource) do
    Map.has_key?(@permissions, resource)
  end

  defp valid_action_for_resource?(resource, action) do
    case Map.get(@permissions, resource) do
      nil -> false
      actions -> action in actions
    end
  end
end
