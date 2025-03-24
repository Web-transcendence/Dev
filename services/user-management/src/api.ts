import fastify from 'fastify'
import userRoutes from "./routes.js"
import websocketPlugin, {WebSocket} from "@fastify/websocket"

const app = fastify();

app.register(websocketPlugin);
app.register(userRoutes);

app.get('/ws-connexion', {websocket: true}, async (connection: WebSocket, req) => {
    console.log("WebS    ocket connection established");

    try {
        connection.on('message', (message: string) => {
            console.log("Received message:", message.toString());
        });


        connection.on('close', () => {
            console.log("WebSocket connection closed by client.");
        });

        connection.on('error', (err) => {
            console.error("WebSocket error:", err);
        });
    } catch (err) {
        console.log(err)
    }
})

const connectedUsers = new Map<string, WebSocket>();

app.listen({port: 5000, host: '0.0.0.0'}, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}`)
})