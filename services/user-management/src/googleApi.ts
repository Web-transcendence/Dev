import {FastifyReply, FastifyRequest} from "fastify";
import { OAuth2Client } from 'google-auth-library';
import { Client_db} from "./User.js";
import jwt from "jsonwebtoken";
import {string} from "zod";

// Initialisation du client Google OAuth2
const client = new OAuth2Client("562995219569-0icrl4jh4ku3h312qmjm8ek57fqt7fp5.apps.googleusercontent.com");

export async function googleAuth(request: FastifyRequest, reply: FastifyReply):Promise<void> {
    const { credential } = request.body as { credential: string };
    try {
        // Vérification du jeton Google
        const ticket = await client.verifyIdToken({
            idToken: credential,
            audience: "562995219569-0icrl4jh4ku3h312qmjm8ek57fqt7fp5.apps.googleusercontent.com",
        });

        // Récupération des informations de l'utilisateur
        const payload = ticket.getPayload();
        const userId = payload?.sub; // ID unique de l'utilisateur

        if (!payload || !userId) {
            throw new Error('Invalid payload from Google');
        }

        // Log des informations de l'utilisateur
        console.log('User ID:', userId);
        console.log('Email:', payload.email);
        console.log('Name:', payload.name);

        if (Client_db.prepare("SELECT * FROM Client WHERE email = ?").get(payload.email)) {
            console.log('Email Already Register:', payload.email);

            return reply.send({token, valid: true, name: payload.name});
        }
        else {
            const res = Client_db.prepare("INSERT INTO Client (name, email, password) VALUES (?, ?, ?)")
                .run(payload.name, payload.email, '');
        }
        // Réponse réussie
        const token = jwt.sign({ name: payload.name, email: payload.email, userId: payload.sub }, 'secret_key', { expiresIn: '1h' });
        // const token = jwt.sign(payload.name, 'secret_key', { expiresIn: '1h' });
        return reply.send({token, valid: true, name: payload.name});
    } catch (error) {
        console.log('Error verifying Google token:', error);
        reply.status(400).send({ success: false, error: 'Invalid token' });
    }
}