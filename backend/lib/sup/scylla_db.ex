defmodule Sup.ScyllaDB do
  @moduledoc """
  ScyllaDB adapter for high-throughput message storage and retrieval.
  """

  use GenServer
  require Logger

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def insert_message(message) do
    GenServer.call(__MODULE__, {:insert_message, message})
  end

  def get_room_messages(room_id, limit \\ 50, before_timestamp \\ nil) do
    GenServer.call(__MODULE__, {:get_room_messages, room_id, limit, before_timestamp})
  end

  def get_message(message_id) do
    GenServer.call(__MODULE__, {:get_message, message_id})
  end

  def search_messages(room_ids, query, limit \\ 20) do
    GenServer.call(__MODULE__, {:search_messages, room_ids, query, limit})
  end

  def update_message(message_id, updates) do
    GenServer.call(__MODULE__, {:update_message, message_id, updates})
  end

  def delete_message(message_id) do
    GenServer.call(__MODULE__, {:delete_message, message_id})
  end

  # Thread support functions
  def get_thread_messages(thread_id, limit \\ 50, before_timestamp \\ nil) do
    GenServer.call(__MODULE__, {:get_thread_messages, thread_id, limit, before_timestamp})
  end

  def count_thread_messages(thread_id) do
    GenServer.call(__MODULE__, {:count_thread_messages, thread_id})
  end

  def get_thread_participants(thread_id) do
    GenServer.call(__MODULE__, {:get_thread_participants, thread_id})
  end

  def search_mentions(user_id, room_ids, limit \\ 20) do
    GenServer.call(__MODULE__, {:search_mentions, user_id, room_ids, limit})
  end

  # GenServer callbacks
  @impl true
  def init(_opts) do
    nodes = Application.get_env(:sup, :scylla_nodes, ["127.0.0.1:9042"])
    keyspace = Application.get_env(:sup, :scylla_keyspace, "sup")

    cluster_opts = [
      nodes: nodes,
      pool_size: 10,
      default_consistency: :one
    ]

    case Xandra.Cluster.start_link(cluster_opts) do
      {:ok, cluster} ->
        # Create keyspace and tables if they don't exist
        setup_keyspace_and_tables(cluster, keyspace)

        Logger.info("Connected to ScyllaDB cluster")
        {:ok, %{cluster: cluster, keyspace: keyspace}}

      {:error, reason} ->
        Logger.error("Failed to connect to ScyllaDB: #{inspect(reason)}")
        {:stop, reason}
    end
  end

  @impl true
  def handle_call({:insert_message, message}, _from, state) do
    query = """
    INSERT INTO #{state.keyspace}.messages
    (id, room_id, sender_id, content, type, timestamp, reply_to_id, encrypted_content)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """

    params = [
      message.id,
      message.room_id,
      message.sender_id,
      message.content,
      Atom.to_string(message.type),
      message.timestamp,
      Map.get(message, :reply_to_id),
      Map.get(message, :encrypted_content)
    ]

    case Xandra.Cluster.execute(state.cluster, query, params) do
      {:ok, _result} ->
        # Also insert into room_messages table for efficient room-based queries
        room_query = """
        INSERT INTO #{state.keyspace}.room_messages
        (room_id, timestamp, message_id, sender_id, content, type)
        VALUES (?, ?, ?, ?, ?, ?)
        """

        room_params = [
          message.room_id,
          message.timestamp,
          message.id,
          message.sender_id,
          message.content,
          Atom.to_string(message.type)
        ]

        case Xandra.Cluster.execute(state.cluster, room_query, room_params) do
          {:ok, _result} ->
            {:reply, :ok, state}

          {:error, reason} ->
            Logger.error("Failed to insert into room_messages: #{inspect(reason)}")
            {:reply, {:error, reason}, state}
        end

      {:error, reason} ->
        Logger.error("Failed to insert message: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:get_room_messages, room_id, limit, before_timestamp}, _from, state) do
    {query, params} =
      case before_timestamp do
        nil ->
          query = """
          SELECT message_id, sender_id, content, type, timestamp
          FROM #{state.keyspace}.room_messages
          WHERE room_id = ?
          ORDER BY timestamp DESC
          LIMIT ?
          """

          {query, [room_id, limit]}

        timestamp ->
          query = """
          SELECT message_id, sender_id, content, type, timestamp
          FROM #{state.keyspace}.room_messages
          WHERE room_id = ? AND timestamp < ?
          ORDER BY timestamp DESC
          LIMIT ?
          """

          {query, [room_id, timestamp, limit]}
      end

    case Xandra.Cluster.execute(state.cluster, query, params) do
      {:ok, %Xandra.Page{} = page} ->
        messages =
          Enum.map(page, fn row ->
            %{
              id: row["message_id"],
              sender_id: row["sender_id"],
              content: row["content"],
              type: String.to_existing_atom(row["type"]),
              timestamp: row["timestamp"]
            }
          end)

        {:reply, {:ok, messages}, state}

      {:error, reason} ->
        Logger.error("Failed to get room messages: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:get_message, message_id}, _from, state) do
    query = """
    SELECT id, room_id, sender_id, content, type, timestamp, reply_to_id
    FROM #{state.keyspace}.messages
    WHERE id = ?
    """

    case Xandra.Cluster.execute(state.cluster, query, [message_id]) do
      {:ok, %Xandra.Page{} = page} ->
        case Enum.to_list(page) do
          [row] ->
            message = %{
              id: row["id"],
              room_id: row["room_id"],
              sender_id: row["sender_id"],
              content: row["content"],
              type: String.to_existing_atom(row["type"]),
              timestamp: row["timestamp"],
              reply_to_id: row["reply_to_id"]
            }

            {:reply, {:ok, message}, state}

          [] ->
            {:reply, {:error, :not_found}, state}
        end

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:search_messages, room_ids, query_text, limit}, _from, state) do
    # Basic text search - in production, you'd use a proper search index
    search_query = """
    SELECT message_id, room_id, sender_id, content, type, timestamp
    FROM #{state.keyspace}.room_messages
    WHERE room_id IN ? AND content LIKE ?
    LIMIT ?
    """

    search_pattern = "%#{query_text}%"

    case Xandra.Cluster.execute(state.cluster, search_query, [room_ids, search_pattern, limit]) do
      {:ok, %Xandra.Page{} = page} ->
        messages =
          Enum.map(page, fn row ->
            %{
              id: row["message_id"],
              room_id: row["room_id"],
              sender_id: row["sender_id"],
              content: row["content"],
              type: String.to_existing_atom(row["type"]),
              timestamp: row["timestamp"]
            }
          end)

        {:reply, {:ok, messages}, state}

      {:error, reason} ->
        Logger.error("Failed to search messages: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:update_message, message_id, updates}, _from, state) do
    # Build dynamic update query based on provided updates
    set_clauses = Enum.map(updates, fn {key, _value} -> "#{key} = ?" end)
    set_clause = Enum.join(set_clauses, ", ")

    query = """
    UPDATE #{state.keyspace}.messages
    SET #{set_clause}
    WHERE id = ?
    """

    params = Map.values(updates) ++ [message_id]

    case Xandra.Cluster.execute(state.cluster, query, params) do
      {:ok, _result} ->
        {:reply, :ok, state}

      {:error, reason} ->
        Logger.error("Failed to update message: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:delete_message, message_id}, _from, state) do
    query = """
    DELETE FROM #{state.keyspace}.messages
    WHERE id = ?
    """

    case Xandra.Cluster.execute(state.cluster, query, [message_id]) do
      {:ok, _result} ->
        {:reply, :ok, state}

      {:error, reason} ->
        Logger.error("Failed to delete message: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:get_thread_messages, thread_id, limit, before_timestamp}, _from, state) do
    {query, params} = if before_timestamp do
      query = """
      SELECT id, room_id, sender_id, content, type, timestamp, reply_to_id
      FROM #{state.keyspace}.thread_messages
      WHERE thread_id = ? AND timestamp < ?
      ORDER BY timestamp DESC
      LIMIT ?
      """
      {query, [thread_id, before_timestamp, limit]}
    else
      query = """
      SELECT id, room_id, sender_id, content, type, timestamp, reply_to_id
      FROM #{state.keyspace}.thread_messages
      WHERE thread_id = ?
      ORDER BY timestamp DESC
      LIMIT ?
      """
      {query, [thread_id, limit]}
    end

    case Xandra.Cluster.execute(state.cluster, query, params) do
      {:ok, %Xandra.Page{} = page} ->
        messages =
          Enum.map(page, fn row ->
            %{
              id: row["id"],
              room_id: row["room_id"],
              sender_id: row["sender_id"],
              content: row["content"],
              type: String.to_existing_atom(row["type"]),
              timestamp: row["timestamp"],
              reply_to_id: row["reply_to_id"]
            }
          end)

        {:reply, {:ok, messages}, state}

      {:error, reason} ->
        Logger.error("Failed to get thread messages: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:count_thread_messages, thread_id}, _from, state) do
    query = """
    SELECT COUNT(*) as count
    FROM #{state.keyspace}.thread_messages
    WHERE thread_id = ?
    """

    case Xandra.Cluster.execute(state.cluster, query, [thread_id]) do
      {:ok, %Xandra.Page{} = page} ->
        count = Enum.at(page, 0)["count"] || 0
        {:reply, {:ok, count}, state}

      {:error, reason} ->
        Logger.error("Failed to count thread messages: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:get_thread_participants, thread_id}, _from, state) do
    query = """
    SELECT DISTINCT sender_id
    FROM #{state.keyspace}.thread_messages
    WHERE thread_id = ?
    """

    case Xandra.Cluster.execute(state.cluster, query, [thread_id]) do
      {:ok, %Xandra.Page{} = page} ->
        participants = Enum.map(page, fn row -> row["sender_id"] end)
        {:reply, {:ok, participants}, state}

      {:error, reason} ->
        Logger.error("Failed to get thread participants: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:search_mentions, user_id, room_ids, limit}, _from, state) do
    query = """
    SELECT id, room_id, sender_id, content, type, timestamp
    FROM #{state.keyspace}.messages
    WHERE room_id IN ? AND content LIKE ?
    ORDER BY timestamp DESC
    LIMIT ?
    """

    mention_pattern = "%@#{user_id}%"

    case Xandra.Cluster.execute(state.cluster, query, [room_ids, mention_pattern, limit]) do
      {:ok, %Xandra.Page{} = page} ->
        messages =
          Enum.map(page, fn row ->
            %{
              id: row["id"],
              room_id: row["room_id"],
              sender_id: row["sender_id"],
              content: row["content"],
              type: String.to_existing_atom(row["type"]),
              timestamp: row["timestamp"]
            }
          end)

        {:reply, {:ok, messages}, state}

      {:error, reason} ->
        Logger.error("Failed to search mentions: #{inspect(reason)}")
        {:reply, {:error, reason}, state}
    end
  end

  # Private functions
  defp setup_keyspace_and_tables(cluster, keyspace) do
    # Create keyspace
    create_keyspace = """
    CREATE KEYSPACE IF NOT EXISTS #{keyspace}
    WITH REPLICATION = {
      'class': 'SimpleStrategy',
      'replication_factor': 3
    }
    """

    Xandra.Cluster.execute(cluster, create_keyspace)

    # Create messages table
    create_messages_table = """
    CREATE TABLE IF NOT EXISTS #{keyspace}.messages (
      id UUID PRIMARY KEY,
      room_id UUID,
      sender_id UUID,
      content TEXT,
      type TEXT,
      timestamp TIMESTAMP,
      reply_to_id UUID,
      encrypted_content BLOB,
      encryption_key_id TEXT,
      edited_at TIMESTAMP
    )
    """

    Xandra.Cluster.execute(cluster, create_messages_table)

    # Create room_messages table (partitioned by room_id for efficient queries)
    create_room_messages_table = """
    CREATE TABLE IF NOT EXISTS #{keyspace}.room_messages (
      room_id UUID,
      timestamp TIMESTAMP,
      message_id UUID,
      sender_id UUID,
      content TEXT,
      type TEXT,
      PRIMARY KEY (room_id, timestamp, message_id)
    ) WITH CLUSTERING ORDER BY (timestamp DESC)
    """

    Xandra.Cluster.execute(cluster, create_room_messages_table)

    # Create thread_messages table (for threaded conversations)
    create_thread_messages_table = """
    CREATE TABLE IF NOT EXISTS #{keyspace}.thread_messages (
      thread_id UUID,
      timestamp TIMESTAMP,
      id UUID,
      room_id UUID,
      sender_id UUID,
      content TEXT,
      type TEXT,
      reply_to_id UUID,
      PRIMARY KEY (thread_id, timestamp, id)
    ) WITH CLUSTERING ORDER BY (timestamp DESC)
    """

    Xandra.Cluster.execute(cluster, create_thread_messages_table)

    Logger.info("ScyllaDB keyspace and tables created/verified")
  end
end
