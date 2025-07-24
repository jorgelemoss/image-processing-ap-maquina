sudo docker compose up --build
if [ $? -eq 0 ]; then
    echo "🚀 Container 'backend-container' iniciado em segundo plano na porta 4000!"
    echo "🚀 Container 'frontend-container' iniciado em segundo plano na porta 3000!"

    echo "ℹ️ Use 'docker logs frontend-container ou backend-container' para ver os logs."
    echo "ℹ️ Use docker ps -a para ver todos os containers rodando"
else
    echo "❌ Falha ao iniciar os containers."
    exit 1
fi


echo ""
echo "✅ O frontend e backend foram gerados"
echo ""