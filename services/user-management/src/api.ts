import fastify from 'fastify'
import userRoutes from "./routes.js"
import {googleAuth} from "./googleApi.js";
import { FastifySSEPlugin } from "fastify-sse-v2";
import { FastifyReply } from "fastify";
import {tournament} from "./tounament.js";

const app = fastify();

app.register(FastifySSEPlugin);
app.register(userRoutes);
app.post('/auth/google', googleAuth);

export const connectedUsers = new Map<string, FastifyReply>();

export const tournamentSessions = new Map<string, tournament>();

app.listen({port: 5000, host: '0.0.0.0'}, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}`)
})