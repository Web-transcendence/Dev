import {FastifyReply, FastifyRequest} from "fastify";
import { OAuth2Client } from 'google-auth-library';

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

        // Réponse réussie
        reply.status(200).send({ success: true, user: payload });
    } catch (error) {
        console.log('Error verifying Google token:', error);
        reply.status(400).send({ success: false, error: 'Invalid token' });
    }
}