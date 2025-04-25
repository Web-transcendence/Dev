import fastify from 'fastify'
import userRoutes from "./routes.js"
import {googleAuth} from "./googleApi.js";
import { FastifySSEPlugin } from "fastify-sse-v2";
import { FastifyReply } from "fastify";
import {tournament} from "./tournament.js";

const app = fastify();

app.register(FastifySSEPlugin);
app.register(userRoutes);
app.post('/auth/google', googleAuth);

export const connectedUsers = new Map<number, FastifyReply>();

export const tournamentSessions = new Map<number, tournament>();
try {
    tournamentSessions.set(4, new tournament(4))
    tournamentSessions.set(8, new tournament(8))
    tournamentSessions.set(16, new tournament(16))
    tournamentSessions.set(32, new tournament(32))
} catch (err) {
    console.error(err);
}



app.listen({port: 5000, host: '0.0.0.0'}, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}`)
})