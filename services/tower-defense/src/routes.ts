import {MatchResult, getMatchHistory} from "./database.js";
import {FastifyInstance, FastifyReply, FastifyRequest} from "fastify";
import {generateRoom} from "./netcode.js";

export default async function tdRoutes(fastify: FastifyInstance) {
    fastify.post('/generateRoom', async (req: FastifyRequest, res: FastifyReply) => {
        const roomId = generateRoom();
        return (res.status(200).send({roomId: roomId}));
    })

    fastify.get('/getMatchHistory', async (req: FastifyRequest, res: FastifyReply) => {
        try {
            const id = Number(req.headers.id);
            const MatchResult: MatchResult[] = getMatchHistory(id);
            return (res.status(200).send(MatchResult));
        } catch (error) {
            console.log(error);
            return (res.status(500).send({error}));
        }
    })
}