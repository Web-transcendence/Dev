all: up

up:
	docker-compose -f sources/docker-compose.yml up --build

detach:
	docker-compose -f sources/docker-compose.yml build

watch: detach
	docker-compose -f sources/docker-compose.yml up --watch

build:
	docker-compose -f sources/docker-compose.yml build

down:
	docker-compose -f sources/docker-compose.yml down

start:
	docker-compose -f sources/docker-compose.yml start

stop:
	docker-compose -f sources/docker-compose.yml stop

logs:
	docker-compose -f sources/docker-compose.yml logs --follow

prune:
	docker system prune --all --volumes --force

mysql:
	docker-compose -f sources/docker-compose.yml exec mariadb mysql

clean:
	docker-compose -f sources/docker-compose.yml down --volumes --rmi all

fclean: clean
#	Use docker run to remove data because of permissions
	docker run -it --rm -v $(HOME)/data:/data busybox sh -c "rm -rf /data/*"

help:
	@echo "Makefile for Docker Compose"
	@echo "Available targets:"
	@echo "  up      - Start services"
	@echo "  build   - Build services"
	@echo "  down    - Remove services"
	@echo "  start   - Start services"
	@echo "  stop    - Stop services"
	@echo "  logs    - View logs"
	@echo "  prune   - Remove all unused containers and images"
	@echo "  mysql   - Execute mariadb monitor"
	@echo "  re      - Restart services with fclean & up"
	@echo "  fclean  - Call clean and remove data, secrets & certificates"
	@echo "  clean   - Remove volumes and stop services"
	@echo "  help    - Show this help message"

.PHONY: all up build down start stop logs prune mysql re fclean clean