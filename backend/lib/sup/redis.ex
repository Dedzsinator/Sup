defmodule Sup.Redis do
  @moduledoc """
  Redis adapter for sessions, caching, and real-time state.
  """

  use GenServer
  require Logger

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def command(command) do
    GenServer.call(__MODULE__, {:command, command})
  end

  def pipeline(commands) do
    GenServer.call(__MODULE__, {:pipeline, commands})
  end

  # Convenience functions for common operations
  def get(key) do
    command(["GET", key])
  end

  def set(key, value) do
    command(["SET", key, value])
  end

  def setex(key, ttl, value) do
    command(["SETEX", key, ttl, value])
  end

  def del(key) do
    command(["DEL", key])
  end

  def exists(key) do
    command(["EXISTS", key])
  end

  def hget(key, field) do
    command(["HGET", key, field])
  end

  def hset(key, field, value) do
    command(["HSET", key, field, value])
  end

  def hgetall(key) do
    command(["HGETALL", key])
  end

  def sadd(key, member) do
    command(["SADD", key, member])
  end

  def srem(key, member) do
    command(["SREM", key, member])
  end

  def smembers(key) do
    command(["SMEMBERS", key])
  end

  def sismember(key, member) do
    command(["SISMEMBER", key, member])
  end

  # GenServer callbacks
  @impl true
  def init(_opts) do
    redis_url = Application.get_env(:sup, :redis_url, "redis://localhost:6379")

    case Redix.start_link(redis_url) do
      {:ok, conn} ->
        Logger.info("Connected to Redis")
        {:ok, %{conn: conn}}

      {:error, reason} ->
        Logger.error("Failed to connect to Redis: #{inspect(reason)}")
        {:stop, reason}
    end
  end

  @impl true
  def handle_call({:command, command}, _from, %{conn: conn} = state) do
    case Redix.command(conn, command) do
      {:ok, result} ->
        {:reply, {:ok, result}, state}

      {:error, reason} ->
        Logger.error("Redis command failed: #{inspect(command)}, reason: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:pipeline, commands}, _from, %{conn: conn} = state) do
    case Redix.pipeline(conn, commands) do
      {:ok, results} ->
        {:reply, {:ok, results}, state}

      {:error, reason} ->
        Logger.error("Redis pipeline failed: #{inspect(commands)}, reason: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def terminate(reason, %{conn: conn}) do
    Logger.info("Redis connection terminating: #{inspect(reason)}")
    GenServer.stop(conn)
  end
end
