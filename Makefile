all: up

generate-key:
	./generate_key.sh
up: generate-key
	docker compose -f docker-compose.yml up --build

continue:
	docker compose -f docker-compose.yml up

build: generate-key
	docker compose -f docker-compose.yml build

down:
	docker compose -f docker-compose.yml down

logs:
	docker compose -f docker-compose.yml logs --follow

prune:
	docker system prune --all --volumes --force

clean:
	@test -f .env || touch .env
	docker compose -f docker-compose.yml down --volumes --rmi all
	@rm -f .env
	@rm -rf data


fclean: clean
	docker run -it --rm -v $(HOME)/data:/data busybox sh -c "rm -rf /data/*"

re: fclean up

full: fclean build


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