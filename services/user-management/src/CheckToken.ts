import fastify from 'fastify'
import { FastifyInstance } from 'fastify';
import {FastifyReply, FastifyRequest} from "fastify";
import jwt from 'jsonwebtoken';

export async function checkToken(fastify: FastifyInstance):Promise<void> {
    fastify.post('/verify-token', async (request, reply) => {
        const {token: string} = request.body;
        if (!token) {
            return reply.status(401).send({valid: false, message: "Token manquant"});
        }

        try {
            const decoded = jwt.verify(token, 'secret_key');
            return reply.send({valid: true, username: (decoded as any).username});
        } catch (error) {
            return reply.status(401).send({valid: false, message: "Token invalide ou expiré"});
        }
    }
});