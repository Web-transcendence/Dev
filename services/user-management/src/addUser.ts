import {FastifyReply, FastifyRequest} from "fastify";


export async function addUser(req: FastifyRequest, res: FastifyReply):Promise<void> {
    const { name, email, password } = req.body as { name: string; email: string; password: string };
    try {
        const response = await fetch(`http://database:4001/email-existing?email=${encodeURIComponent(email)}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        if (!response.ok) {
            return res.status(400).send({ error: "Email déjà utilisé" });
        }
        // let hashedPassword = '';
        // if (password) {
        //     hashedPassword = await bcrypt.hash(password, 10);
        // }
        const addUserRes = await fetch('http://database:4001/addUser', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                name: name,
                email: email,
                password: password
                // password: hashedPassword
            })
        });
        if (!addUserRes.ok) {
            const errorData = await addUserRes.json();
            return res.status(500).send({ json: errorData});
        }
        res.status(201).send({redirect: `/part/login?name=${encodeURIComponent("User")}`});
    } catch (err) {
        console.error(err);
    }
}