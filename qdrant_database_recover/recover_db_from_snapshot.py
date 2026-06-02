from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:3336", timeout=10000)

# Путь ВНУТРИ контейнера (соответствует вашему -v маунту)
# Файл лежит в root примонтированной папки /qdrant/storage/
snapshot_location = "file:///qdrant/snapshots/patents_collection_bge-8618925416111929-2026-05-18-18-41-28.snapshot"

print(f"🔄 Восстанавливаю из {snapshot_location}...")

try:
    client.recover_snapshot(
        collection_name="patents_collection_bge",
        location=snapshot_location
    )
    print("Успешно восстановлено!")

    # Проверка: получаем информацию о коллекции
    info = client.get_collection("patents_collection_bge")
    print(f"Коллекция содержит {info.vectors_count} векторов")

except Exception as e:
    print(f"Ошибка: {e}")