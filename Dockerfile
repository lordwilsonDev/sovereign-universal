# Sovereign Hub/API Dockerfile
FROM node:20-slim

WORKDIR /app

COPY package*.json ./
RUN npm install --production

COPY . .

EXPOSE 8000
CMD ["node", "api/server.js"]
