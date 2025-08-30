from search import search_prompt

def main():
    run = search_prompt()
    print("Faça sua pergunta (digite 'sair' para encerrar):\n")

    while True:
        pergunta = input("PERGUNTA: ").strip()
        if not pergunta or pergunta.lower() in ("sair", "exit", "quit"):
            print("Encerrando. Até mais!")
            break

        resposta = run(pergunta)
        print(f"\nRESPOSTA: {resposta}\n")

if __name__ == "__main__":
    main()
