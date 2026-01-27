# Configuração do Email para Envio de PDFs

## ⚙️ Configuração Necessária

1. **Ativar a verificação em duas etapas no Gmail**
   - Acesse: https://myaccount.google.com/security
   - Clique em "Verificação em duas etapas"
   - Siga as instruções para ativar

2. **Gerar uma Senha de App**
   - Acesse: https://myaccount.google.com/apppasswords
   - Digite o nome do app (ex: "LLMs Project")
   - Clique em "Criar"
   - Copie a senha gerada (16 caracteres)

3. **Atualizar o arquivo `.env`**
   ```properties
   EMAIL_PASSWORD=sua_senha_de_app_aqui
   REMETENTE=seu_email@gmail.com
   ```

