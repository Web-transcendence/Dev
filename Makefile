all: up

up:
	docker compose -f docker-compose.yml up --build

watch:
	docker compose -f docker-compose.yml up --watch

build:
	docker compose -f docker-compose.yml build

down:
	docker compose -f docker-compose.yml down

logs:
	docker compose -f docker-compose.yml logs --follow

prune:
	docker system prune --all --volumes --force

mysql:
	docker compose -f docker-compose.yml exec mariadb mysql

clean:
	docker compose -f docker-compose.yml down --volumes --rmi all

fclean: clean
#	Use docker run to remove data because of permissions
	docker run -it --rm -v $(HOME)/data:/data busybox sh -c "rm -rf /data/*"

re: fclean up

rewatch: fclean watch

bw: build watch

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