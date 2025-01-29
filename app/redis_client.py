import redis

# Example configuration — adapt to your own credentials/hostname
# Make sure a Redis server is running and accessible
redis_cli = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True,  # so we can get/set strings directly
)
