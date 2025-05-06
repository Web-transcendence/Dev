import fastify from 'fastify'
import tournamentRoutes from "./routes.js"
import {tournament} from "./tournament.js";

export const INTERNAL_PASSWORD = process.env.SECRET_KEY;

const app = fastify();

app.register(tournamentRoutes);
export const tournamentSessions = new Map<number, tournament>();
try {
    tournamentSessions.set(4, new tournament(4))
    tournamentSessions.set(8, new tournament(8))
    tournamentSessions.set(16, new tournament(16))
    tournamentSessions.set(32, new tournament(32))
} catch (err) {
    console.error(err);
}



app.listen({port: 7000, host: '0.0.0.0'}, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}`)
})