import {FastifyReply, FastifyRequest} from 'fastify';
import jwt from 'jsonwebtoken';

interface decodedToken {
    id: string,
    name: string,
    iat: number,
    exp: number,
}

export async function checkToken(request: FastifyRequest, reply: FastifyReply):Promise<void> {
    const {token} = request.body as { token: string };
    if (!token) {
        return reply.status(401).send({valid: false, message: "Token manquant"});
    }
    try {
        const decoded = jwt.verify(token, 'secret_key');
        const finder_result = await fetch('http://database:4001/findUser', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                token: decoded,
            })
        });
        const {name} = decoded as decodedToken;
        if (finder_result && name) {
            return reply.send({valid: true, username: (name)});
        }
        else
            return reply.send({valid: false, message: "Invalid name"});
    } catch (error) {
        return reply.status(401).send({valid: false, message: "Token invalide ou expiré"});
    }
}

export async function checkUser(request: FastifyRequest, reply: FastifyReply):Promise<void> {
    const {token} = request.body as { token: string };
    if (!token) {
        return reply.status(401).send({valid: false, message: "Token manquant"});
    }
    try {
        const decoded = jwt.verify(token, 'secret_key');
        const finder_result = await fetch('http://database:4001/checkUser', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                token: decoded,
            })
        });
        const {name} = decoded as decodedToken;
        if (finder_result && name) {
            return reply.send({valid: true, username: (name)});
        }
        else
            return reply.send({valid: false, message: "Invalid name"});
    } catch (error) {
        return reply.status(401).send({valid: false, message: "Token invalide ou expiré"});
    }
}