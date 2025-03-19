import fastify from 'fastify';
import { WebSocketServer } from 'ws';
import http from 'http';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = fastify();

const server = http.createServer(app.server);
const wss = new WebSocketServer({ server });

app.register(import('@fastify/static'), {
    root: path.join(__dirname, '../public'),
    prefix: '/public/',
});

// WebSocket connection handler
wss.on('connection', (ws) => {
    console.log('Client connected');
    ws.on('message', (message) => {
        const data = JSON.parse(message.toString());
        if (data.type === 'input') {
            console.log(`Touche ${data.key} est ${data.state}`);
            // Traiter le mouvement de la raquette ici
        }
    });
    ws.on('close', () => console.log('Client disconnected'));
});

app.listen({port: 8080, host: '0.0.0.0'}, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}`)
})