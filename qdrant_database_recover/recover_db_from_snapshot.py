
import qdrant_client

client = QdrantClient(url="http://localhost:6333", timeout=900)


snapshot_name = "patents_collection_snapshot.snapshot"
print(f"🔄 Восстанавливаю из {snapshot_name}...")

try:
    # Восстанавливаем коллекцию из снапшота
    client.recover_snapshot(
        collection_name="patents_collection_2",
        location="file:///qdrant/snapshots/patents_collection_snapshot.snapshot"
    )
    print("Успешно восстановлено!")

    # Проверка: получаем информацию о коллекции
    info = client.get_collection("patents_collection_2")
    print(f"Коллекция содержит {info.vectors_count} векторов")

except Exception as e:
    print(f"Ошибка: {e}")