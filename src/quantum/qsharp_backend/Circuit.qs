namespace QuantumClassifier {
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Measurement;

    /// Angle encoding + hardware-efficient ansatz (2 qubits).
    /// Por capa: Ry(θ)Rz(φ) por qubit + CNOT(0→1).
    /// Retorna expectativa <Z0> estimada via simulación.
    operation EncodeAndAnsatz(
        x : Double[],
        theta : Double[],
        nLayers : Int,
        qubits : Qubit[]
    ) : Unit {
        // encoding
        for i in 0..1 {
            Ry(x[i], qubits[i]);
        }
        // ansatz
        mutable idx = 0;
        for layer in 0..nLayers-1 {
            for q in 0..1 {
                Ry(theta[idx], qubits[q]);
                Rz(theta[idx+1], qubits[q]);
                set idx += 2;
            }
            CNOT(qubits[0], qubits[1]);
        }
    }

    operation EstimateExpectationZ(
        x : Double[],
        theta : Double[],
        nLayers : Int
    ) : Double {
        use qubits = Qubit[2];
        EncodeAndAnsatz(x, theta, nLayers, qubits);
        // Medición en base Z del qubit 0 sin colapsar vía Dump no disponible aquí;
        // en simulador se estima repitiendo. Para paridad, el wrapper Python
        // compara contra el simulador numpy usando el mismo circuito unitario.
        // Esta operación existe como artefacto documentado y punto de entrada
        // para `qsharp` vía Python: qsharp.compile + simulate.
        let r = MResetZ(qubits[0]);
        Reset(qubits[1]);
        return r == Zero ? 1.0 | -1.0;
    }
}
