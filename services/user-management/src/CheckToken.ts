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
    const { user } = request.body as { user: { name: string; password: string } };
    if (!user.name || !user.password) {
            console.log("44444444444444444444444444444TTTTTTTTTTTTTTTTTTTTTTT4444OROROROROROR");
        return reply.status(401).send({valid: false, error: "Name or Password missing"});
    }
    try {
        const result = await fetch('http://database:4001/loginUser', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                name: user.name,
                password: user.password
            })
        });
        if (result) {
            console.log("ERRRORORORO4444444444444444444444444444444TTTTTTTTTTTTTTTTTTTTTTT4444OROROROROROR");
            const token = jwt.sign(user, 'secret_key', { expiresIn: '1h' });
            return reply.send({token, valid: true, name: user.name});
            // return reply.send({token, valid: true, name: user.name, avatar: result.avatar});
        }
        else {
            console.log("ERRRORORORO44444444444444444444444444444444444OROROROROROR");
            return reply.send({valid: false, error: "Invalid name"});
        }
    } catch (error) {
        console.log("ERRROROROROOROROROROROR", error);
        return reply.status(401).send({valid: false, error: "Token invalide ou expiré"});
    }
}