import express from "express";
import { WebSocketServer } from "ws";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const wss = new WebSocketServer({ port:8080});

app.use(express.static(path.join(__dirname, "../public")));

wss.on("connection", (ws) => {
    console.log("Client connected");
    ws.on("message", (message) => {
        const data = JSON.parse(message.toString());
        if (data.type === "input") {
            console.log(`Touche ${data.key} est ${data.state}`);
            // Traiter le mouvement de la raquette ici
        }
    });
    ws.on("close", () => console.log("Client disconnected"));
});

console.log("Server running on 8080");