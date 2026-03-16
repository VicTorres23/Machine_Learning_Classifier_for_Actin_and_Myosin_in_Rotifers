nextflow.enable.dsl=2

params.input_fasta = null
params.outdir = nullparams.sequences_per_file = 1000

process SPLIT_FASTA {
    tag "${input_fasta.simpleName}"
    publishDir "${params.outdir}/chunks", mode: 'copy'

    input:
    path input_fasta

    output:
    path "chunk_*.fasta"

    script:
    """
    Split_FASTAS.py \
        --input_fasta ${input_fasta} \
        --output_folder . \
        --sequences_per_file ${params.sequences_per_file}
    """
}

process FASTA_2_CSV {
    tag "${chunk_fasta.baseName}"
    publishDir "${params.outdir}/csv_chunks", mode: 'copy'

    input:
    path chunk_fasta

    output:
    path "${chunk_fasta.baseName}.csv"

    script:
    """
    getCSVFilesFromFASTA.py \
        --input_fasta ${chunk_fasta} \
        --output_csv ${chunk_fasta.baseName}.csv
    """
}

process MERGE_CSVS {
    tag "merge_all_csvs"
    publishDir "{params.outdir}/merged", mode: 'copy'

    input:
    path csv_files

    output:
    path "merged_dataset.csv"

    script:
    """
    mdkir csv_inputs
    cp ${csv_files} csv_inputs/

    MergeAllCSVFiles.py \
        --input_folder csv_inputs \
        --output_file merged_dataset.csv
    """
}

workflow {
    fasta_input = Channel.fromPath(params.input_fasta, checkIfExists: true)
    chunks = SPLIT_FASTA(fasta_input)
    csv_chunks = FASTA_2_CSV(chunks)
    merged = MERGE_CSVS(csv_chunks.collect())
}