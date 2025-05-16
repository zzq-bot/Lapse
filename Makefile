pull:
	@git pull github main
	@echo "pull done"

push:
	@git push origin main
	@git push github main
	@echo "push done"

wc:
	@wc -l main.py models.py utils.py overlap.py agents.py
