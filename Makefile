.PHONY: setup data train score retriever evaluate test api app docker

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

data:
	python3 scripts/generate_customer_data.py --rows 600 --seed 42
	python3 scripts/generate_documents.py --docs-per-customer 4 --seed 42

train:
	python3 scripts/train_churn_model.py

score:
	python3 scripts/score_customers.py

retriever:
	python3 scripts/build_retriever.py

evaluate:
	python3 scripts/evaluate_rag.py

test:
	python3 -m pytest -q

api:
	python3 -m uvicorn src.api.main:app --reload

app:
	python3 -m streamlit run app/streamlit_app.py

docker:
	docker compose up --build

